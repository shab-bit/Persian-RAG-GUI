#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persian RAG GUI — پرسش و پاسخ هوشمند از کتاب‌های PDF پردازش‌شده.

ویژگی‌ها:
  • ذخیره‌سازی خودکار جلسه (بازیابی در اجرای بعدی)
  • افزودن به ایندکس یا بازسازی کامل
  • نمایش همیشگی متن‌های بازیابی‌شده قبل از پاسخ مدل
  • کلید روشن/خاموش کردن LLM (برای مشاهده‌ی متن خالص)
  • پشتیبانی از qwen2.5، deepseek-r1، llama3.1 و ...
  • چیدمان RTL و رابط کاربری کاملاً فارسی
  • پردازش در Thread جداگانه (UI فریز نمی‌شود)

پیش‌نیازها:
    pip install langchain langchain-community langchain-ollama \
                langchain-text-splitters chromadb
    ollama pull nomic-embed-text    # یا bge-m3
    ollama pull qwen2.5:7b          # یا مدل دلخواه
"""

from __future__ import annotations

import json
import queue
import re
import shutil
import threading
import urllib.request
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont

# ─── LangChain (soft import) ────────────────────────────────────────
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_ollama import OllamaEmbeddings, ChatOllama
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_OK = True
    LANGCHAIN_ERR = ""
except ImportError as e:
    LANGCHAIN_OK = False
    LANGCHAIN_ERR = str(e)


# ═══════════════════════════════════════════════════════════════════════
# مسیرها و تنظیمات
# ═══════════════════════════════════════════════════════════════════════

CONFIG_DIR   = Path.home() / ".persian_rag"
CONFIG_FILE  = CONFIG_DIR / "config.json"
SESSION_FILE = CONFIG_DIR / "session.json"


@dataclass
class Settings:
    ollama_url: str      = "http://localhost:11434"
    embed_model: str     = "nomic-embed-text"
    llm_model: str       = "qwen2.5-coder:7b"
    chunk_size: int      = 400
    chunk_overlap: int   = 60
    top_k: int           = 5
    temperature: float   = 0.1
    recursive_scan: bool = False
    use_llm: bool        = True

    @classmethod
    def load(cls) -> "Settings":
        try:
            if CONFIG_FILE.exists():
                data = json.loads(CONFIG_FILE.read_text("utf-8"))
                return cls(**{k: v for k, v in data.items()
                              if k in cls.__dataclass_fields__})
        except Exception:
            pass
        return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


@dataclass
class Session:
    last_folder: str = ""
    recursive: bool = False
    books: list[dict] = field(default_factory=list)
    index_ready: bool = False
    total_chunks: int = 0
    embed_model_at_index: str = ""
    indexed_at: str = ""

    @classmethod
    def load(cls) -> "Session":
        try:
            if SESSION_FILE.exists():
                data = json.loads(SESSION_FILE.read_text("utf-8"))
                return cls(**{k: v for k, v in data.items()
                              if k in cls.__dataclass_fields__})
        except Exception:
            pass
        return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        if SESSION_FILE.exists():
            SESSION_FILE.unlink()
        self.last_folder = ""
        self.books = []
        self.index_ready = False
        self.total_chunks = 0
        self.embed_model_at_index = ""
        self.indexed_at = ""


# ═══════════════════════════════════════════════════════════════════════
# RAG Engine
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Book:
    name: str
    path: Path
    pages: int
    useful_pages: int = 0
    chars: int = 0


def _strip_thinking(text: str) -> str:
    """حذف بلوک‌های  thinking...  از خروجی DeepSeek-R1 و مدل‌های مشابه."""
    text = re.sub(r" thinking.*? response", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s* thinking.*?$", "", text, flags=re.DOTALL)
    return text.strip()


class RAGEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.books: list[Book] = []
        self.vectordb: Chroma | None = None
        self.chain = None
        self.retriever = None
        self.total_chunks = 0
        self.chroma_dir = CONFIG_DIR / "chroma"
        self.session = Session.load()

    # ─── بازگرداندن جلسه ────────────────────────────────────────────
    def restore_session(self) -> tuple[bool, str]:
        s = self.session
        if not s.index_ready:
            return False, "جلسه‌ای ذخیره نشده."
        if not self.chroma_dir.exists():
            return False, "پایگاه داده‌ی برداری پیدا نشد."
        if s.embed_model_at_index != self.settings.embed_model:
            return False, (f"مدل Embedding عوض شده "
                           f"({s.embed_model_at_index} → "
                           f"{self.settings.embed_model})")
        try:
            self.books = [
                Book(
                    name=b["name"],
                    path=Path(b["path"]),
                    pages=b.get("pages", 0),
                    useful_pages=b.get("useful_pages", 0),
                    chars=b.get("chars", 0),
                )
                for b in s.books
            ]
            embeddings = OllamaEmbeddings(
                model=self.settings.embed_model,
                base_url=self.settings.ollama_url,
            )
            self.vectordb = Chroma(
                persist_directory=str(self.chroma_dir),
                embedding_function=embeddings,
                collection_name="persian_rag",
            )
            self.total_chunks = self.vectordb._collection.count()
            self._setup_chain()
            return True, ""
        except Exception as e:
            return False, f"خطا در بازیابی: {e}"

    # ─── ذخیره‌ی جلسه ───────────────────────────────────────────────
    def _save_session(self) -> None:
        self.session.books = [
            {**asdict(b), "path": str(b.path)} for b in self.books
        ]
        self.session.last_folder = (
            str(self.books[0].path.parent) if self.books else "")
        self.session.recursive = self.settings.recursive_scan
        self.session.index_ready = True
        self.session.total_chunks = self.total_chunks
        self.session.embed_model_at_index = self.settings.embed_model
        self.session.indexed_at = datetime.now().isoformat(timespec="seconds")
        self.session.save()

    # ─── اسکن پوشه ──────────────────────────────────────────────────
    def scan_folder(self, folder: Path, recursive: bool = False) -> list[Book]:
        pattern = "**/*.jsonl" if recursive else "*.jsonl"
        files = sorted(folder.glob(pattern))
        books: list[Book] = []
        for f in files:
            info = self._inspect_jsonl(f)
            if info:
                books.append(info)
        self.books = books
        return books

    def _inspect_jsonl(self, path: Path) -> Book | None:
        pages = useful = chars = 0
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    pages += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("is_empty") or rec.get("is_toc"):
                        continue
                    if "error" in rec:
                        continue
                    text = rec.get("text", "").strip()
                    if len(text) >= 50:
                        useful += 1
                        chars += len(text)
        except Exception:
            return None
        return Book(name=path.stem, path=path,
                    pages=pages, useful_pages=useful, chars=chars)

    # ─── ایندکس کردن ────────────────────────────────────────────────
    def build_index(
        self,
        rebuild: bool = False,
        progress_cb=None,
        status_cb=None,
    ) -> None:
        def log(msg):
            if status_cb:
                status_cb(msg)

        # 1) بارگذاری اسناد
        docs: list[Document] = []
        total_books = len(self.books)
        for i, book in enumerate(self.books, 1):
            log(f"خواندن کتاب {i}/{total_books}: {book.name}")
            try:
                with book.path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if rec.get("is_empty") or rec.get("is_toc"):
                            continue
                        if "error" in rec:
                            continue
                        text = rec.get("text", "").strip()
                        if len(text) < 50:
                            continue
                        docs.append(Document(
                            page_content=text,
                            metadata={
                                "book": book.name,
                                "page": rec.get("page"),
                                "method": rec.get("method", "?"),
                            },
                        ))
            except Exception as e:
                log(f"⚠ خطا در {book.name}: {e}")

        if not docs:
            raise RuntimeError("هیچ متن مفیدی برای ایندکس پیدا نشد.")

        # 2) تکه‌تکه کردن
        log(f"تکه‌تکه کردن {len(docs)} صفحه…")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "؟", "؛", "،", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        log(f"ساخته شد: {len(chunks)} تکه")

        # 3) Embedding + Chroma
        log("تولید Embedding … (ممکن است چند دقیقه طول بکشد)")
        embeddings = OllamaEmbeddings(
            model=self.settings.embed_model,
            base_url=self.settings.ollama_url,
        )

        if rebuild and self.chroma_dir.exists():
            shutil.rmtree(self.chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        batch = 64
        vectordb = None

        if not rebuild and (self.chroma_dir / "chroma.sqlite3").exists():
            log("افزودن به ایندکس موجود…")
            vectordb = Chroma(
                persist_directory=str(self.chroma_dir),
                embedding_function=embeddings,
                collection_name="persian_rag",
            )

        for i in range(0, len(chunks), batch):
            end = min(i + batch, len(chunks))
            chunk_batch = chunks[i:end]
            if vectordb is None:
                vectordb = Chroma.from_documents(
                    documents=chunk_batch,
                    embedding=embeddings,
                    persist_directory=str(self.chroma_dir),
                    collection_name="persian_rag",
                )
            else:
                vectordb.add_documents(chunk_batch)
            if progress_cb:
                progress_cb(end, len(chunks))

        self.vectordb = vectordb
        self.total_chunks = self.vectordb._collection.count()
        self._setup_chain()
        self._save_session()
        log(f"✅ ایندکس آماده است. ({self.total_chunks} تکه)")

    # ─── ساخت chain ─────────────────────────────────────────────────
    def _setup_chain(self) -> None:
        s = self.settings
        llm = ChatOllama(
            model=s.llm_model,
            base_url=s.ollama_url,
            temperature=s.temperature,
        )
        self.retriever = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": s.top_k},
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "تو یک دستیار پژوهشی فارسی هستی. تنها بر اساس متن داده‌شده "
             "پاسخ بده. اگر پاسخ در متن نبود، صریحاً بگو که اطلاعات کافی "
             "نداری. پاسخ را کوتاه، دقیق و به فارسی روان بنویس و در صورت "
             "لزوم به شماره صفحه ارجاع بده."),
            ("human",
             "متن زمینه:\n{context}\n\n"
             "سؤال: {question}\n\n"
             "پاسخ:"),
        ])

        def fmt(docs):
            out = []
            for d in docs:
                m = d.metadata
                out.append(
                    f"[{m.get('book','?')} – ص {m.get('page','?')}]\n"
                    f"{d.page_content}"
                )
            return "\n\n---\n\n".join(out)

        self.chain = (
            {"context": self.retriever | fmt,
             "question": lambda x: x}
            | prompt
            | llm
            | StrOutputParser()
        )

    # ─── پرسش ───────────────────────────────────────────────────────
    def query(self, question: str, use_llm: bool = True
              ) -> tuple[str, list[dict]]:
        """بازیابی و (اختیاری) تولید پاسخ.

        Returns
        -------
        (answer, sources)
            answer = "" اگر use_llm=False
            sources = list of dicts با فیلدهای book / page / text
        """
        if not self.retriever:
            raise RuntimeError("ایندکس آماده نیست.")

        # بازیابی تکه‌ها
        docs = self.retriever.invoke(question)
        sources = [
            {
                "book": d.metadata.get("book", "?"),
                "page": d.metadata.get("page", "?"),
                "text": d.page_content,
            }
            for d in docs
        ]

        # اگر LLM خاموش است، فقط متن‌ها را برگردان
        if not use_llm or not self.chain:
            return "", sources

        # تولید پاسخ
        answer = self.chain.invoke(question)
        answer = _strip_thinking(answer)
        return answer, sources


# ═══════════════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════════════

def pick_font() -> str:
    try:
        families = set(tkfont.families())
    except Exception:
        families = set()
    for f in ["Vazirmatn", "Vazir", "IRANSans", "Shabnam",
              "Sahel", "Tahoma", "Arial"]:
        if f in families:
            return f
    return "Tahoma"


def check_ollama(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


class RAGApp:
    BG          = "#f7f7fa"
    SIDEBAR_BG  = "#ffffff"
    ACCENT      = "#2563eb"
    ACCENT_DARK = "#1d4ed8"
    USER_BG     = "#dbeafe"
    BOT_BG      = "#ecfdf5"
    SRC_FG      = "#6b7280"
    TEXT_FG     = "#111827"

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("پرسش و پاسخ هوشمند — RAG فارسی")
        self.root.geometry("1280x820")
        self.root.minsize(980, 640)
        self.root.configure(bg=self.BG)

        self.font_family = pick_font()
        self.settings = Settings.load()
        self.engine = RAGEngine(self.settings)
        self.ui_queue: queue.Queue = queue.Queue()

        try:
            self.root.tk.call("tk", "scaling", 1.2)
        except Exception:
            pass

        if not LANGCHAIN_OK:
            messagebox.showerror(
                "خطای پیش‌نیاز",
                "کتابخانه‌های LangChain نصب نیستند.\n\n"
                "pip install langchain langchain-community "
                "langchain-text-splitters langchain-ollama chromadb\n\n"
                f"جزئیات خطا: {LANGCHAIN_ERR}",
            )
            self.root.destroy()
            return

        self._setup_style()
        self._build_top_bar()
        self._build_main()
        self._build_status_bar()

        self.root.after(100, self._poll_ui_queue)
        self._set_status("آماده. در حال بازیابی جلسه‌ی قبلی…")

        self.root.after(300, self._try_restore_session)

    # ─── استایل ─────────────────────────────────────────────────────
    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=self.BG)
        style.configure("Sidebar.TFrame", background=self.SIDEBAR_BG)
        style.configure("TLabel", background=self.BG,
                        foreground=self.TEXT_FG,
                        font=(self.font_family, 11))
        style.configure("Sidebar.TLabel",
                        background=self.SIDEBAR_BG,
                        foreground=self.TEXT_FG,
                        font=(self.font_family, 11))
        style.configure("Title.TLabel",
                        background=self.BG, foreground=self.ACCENT_DARK,
                        font=(self.font_family, 16, "bold"))
        style.configure("Heading.TLabel",
                        background=self.SIDEBAR_BG, foreground=self.TEXT_FG,
                        font=(self.font_family, 12, "bold"))
        style.configure("Accent.TButton",
                        font=(self.font_family, 11, "bold"),
                        padding=8)
        style.configure("TButton",
                        font=(self.font_family, 10), padding=6)

    # ─── نوار بالا ──────────────────────────────────────────────────
    def _build_top_bar(self):
        bar = tk.Frame(self.root, bg=self.BG, height=64)
        bar.pack(side="top", fill="x", padx=12, pady=(10, 6))
        bar.pack_propagate(False)

        ttk.Label(bar, text="📚  پرسش و پاسخ هوشمند از کتاب‌ها",
                  style="Title.TLabel").pack(side="right", padx=6)

        # چک‌باکس LLM (سمت چپ، کنار دکمه‌ها)
        self.use_llm_var = tk.BooleanVar(value=self.settings.use_llm)
        chk = tk.Checkbutton(
            bar,
            text="🤖  تولید پاسخ با LLM",
            variable=self.use_llm_var,
            command=self._on_use_llm_toggle,
            bg=self.BG, fg=self.ACCENT_DARK,
            activebackground=self.BG,
            activeforeground=self.ACCENT_DARK,
            font=(self.font_family, 11, "bold"),
            selectcolor="#dbeafe",
            anchor="e", justify="right",
            cursor="hand2",
        )
        chk.pack(side="left", padx=(8, 4))

    def _on_use_llm_toggle(self):
        self.settings.use_llm = self.use_llm_var.get()
        self.settings.save()
        if self.use_llm_var.get():
            self._set_status(
                "🤖 LLM روشن است — پاسخ متنی تولید می‌شود.")
            self.send_btn.configure(text="ارسال و پاسخ  ⏎")
        else:
            self._set_status(
                "📚 LLM خاموش است — فقط متن‌های بازیابی‌شده نمایش "
                "داده می‌شوند.")
            self.send_btn.configure(text="جستجو  ⏎")

    # ─── بدنه اصلی ──────────────────────────────────────────────────
    def _build_main(self):
        main = tk.Frame(self.root, bg=self.BG)
        main.pack(side="top", fill="both", expand=True, padx=12, pady=6)

        self.sidebar = ttk.Frame(main, style="Sidebar.TFrame", width=290)
        self.sidebar.pack(side="right", fill="y", padx=(8, 0))
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        self.chat_frame = tk.Frame(main, bg=self.BG)
        self.chat_frame.pack(side="left", fill="both", expand=True,
                             padx=(8, 0))
        self._build_chat()

    def _build_sidebar(self):
        ttk.Label(self.sidebar, text="📖  کتاب‌ها",
                  style="Heading.TLabel").pack(anchor="e", padx=12,
                                               pady=(14, 6))

        list_wrap = tk.Frame(self.sidebar, bg=self.SIDEBAR_BG)
        list_wrap.pack(fill="both", expand=True, padx=12, pady=4)

        self.books_list = tk.Listbox(
            list_wrap,
            font=(self.font_family, 10),
            bg="#fafafa", fg=self.TEXT_FG,
            selectbackground=self.USER_BG,
            selectforeground=self.TEXT_FG,
            activestyle="none",
            highlightthickness=0,
            borderwidth=1, relief="solid",
            justify="right",
        )
        self.books_list.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(list_wrap, orient="vertical",
                           command=self.books_list.yview)
        sb.pack(side="right", fill="y")
        self.books_list.configure(yscrollcommand=sb.set)

        stats_box = tk.Frame(self.sidebar, bg=self.SIDEBAR_BG)
        stats_box.pack(fill="x", padx=12, pady=(10, 4))

        self.stat_books  = ttk.Label(stats_box, text="کتاب‌ها: 0",
                                     style="Sidebar.TLabel")
        self.stat_pages  = ttk.Label(stats_box, text="صفحات مفید: 0",
                                     style="Sidebar.TLabel")
        self.stat_chars  = ttk.Label(stats_box, text="کاراکترها: 0",
                                     style="Sidebar.TLabel")
        self.stat_chunks = ttk.Label(stats_box, text="تکه‌ها: 0",
                                     style="Sidebar.TLabel")
        for w in (self.stat_books, self.stat_pages,
                  self.stat_chars, self.stat_chunks):
            w.pack(anchor="e")

        self.progress = ttk.Progressbar(self.sidebar, mode="determinate",
                                        maximum=100)
        self.progress.pack(fill="x", padx=12, pady=(8, 2))
        self.progress_label = ttk.Label(self.sidebar, text="",
                                        style="Sidebar.TLabel")
        self.progress_label.pack(anchor="e", padx=12)

        btns = tk.Frame(self.sidebar, bg=self.SIDEBAR_BG)
        btns.pack(fill="x", padx=12, pady=(10, 14))

        ttk.Button(btns, text="📂  انتخاب پوشه",
                   command=self.pick_folder,
                   style="Accent.TButton").pack(fill="x", pady=3)

        self.index_btn = ttk.Button(
            btns, text="🔍  افزودن به ایندکس",
            command=lambda: self.start_index(rebuild=False))
        self.index_btn.pack(fill="x", pady=3)

        self.rebuild_btn = ttk.Button(
            btns, text="♻  بازسازی کامل",
            command=lambda: self.start_index(rebuild=True))
        self.rebuild_btn.pack(fill="x", pady=3)

        ttk.Button(btns, text="⚙  تنظیمات",
                   command=self.open_settings).pack(fill="x", pady=3)

        ttk.Button(btns, text="🗑  حذف همه چیز",
                   command=self.clear_books).pack(fill="x", pady=3)

    def _build_chat(self):
        box = tk.Frame(self.chat_frame, bg="#ffffff",
                       highlightthickness=1, highlightbackground="#e5e7eb")
        box.pack(side="top", fill="both", expand=True)

        self.chat_display = tk.Text(
            box, wrap="word",
            font=(self.font_family, 11),
            bg="#ffffff", fg=self.TEXT_FG,
            padx=16, pady=14,
            state="disabled", borderwidth=0,
            spacing1=4, spacing3=8, cursor="arrow",
        )
        self.chat_display.pack(side="left", fill="both", expand=True,
                               padx=(1, 0), pady=1)

        sb = ttk.Scrollbar(box, orient="vertical",
                           command=self.chat_display.yview)
        sb.pack(side="right", fill="y")
        self.chat_display.configure(yscrollcommand=sb.set)

        ff = self.font_family
        # ─── تگ‌های پیام کاربر ────────────────────────────────────
        self.chat_display.tag_configure(
            "user_label", font=(ff, 11, "bold"),
            foreground=self.ACCENT_DARK, justify="right",
            spacing1=12, spacing3=2)
        self.chat_display.tag_configure(
            "user_text", font=(ff, 11),
            foreground=self.TEXT_FG, justify="right",
            lmargin1=40, lmargin2=40, rmargin=10,
            background=self.USER_BG, spacing3=8)

        # ─── تگ‌های بخش منابع ──────────────────────────────────────
        self.chat_display.tag_configure(
            "src_section", font=(ff, 11, "bold"),
            foreground="#7c3aed", justify="right",
            spacing1=10, spacing3=2)
        self.chat_display.tag_configure(
            "src_header", font=(ff, 10, "bold"),
            foreground="#6d28d9", justify="right",
            lmargin1=20, rmargin=10, spacing1=8, spacing3=2)
        self.chat_display.tag_configure(
            "src_text", font=(ff, 10),
            foreground="#374151", justify="right",
            lmargin1=40, lmargin2=40, rmargin=20,
            background="#faf5ff", spacing3=6)
        self.chat_display.tag_configure(
            "src_more", font=(ff, 9, "italic"),
            foreground="#9ca3af", justify="right",
            lmargin1=40, rmargin=20, spacing3=4)

        # ─── تگ‌های پاسخ LLM ──────────────────────────────────────
        self.chat_display.tag_configure(
            "bot_label", font=(ff, 11, "bold"),
            foreground="#047857", justify="right",
            spacing1=12, spacing3=2)
        self.chat_display.tag_configure(
            "bot_text", font=(ff, 11),
            foreground=self.TEXT_FG, justify="right",
            lmargin1=40, lmargin2=40, rmargin=10,
            background=self.BOT_BG, spacing3=6)

        # ─── سایر تگ‌ها ────────────────────────────────────────────
        self.chat_display.tag_configure(
            "info", font=(ff, 10, "italic"),
            foreground="#9ca3af", justify="right", spacing1=6)
        self.chat_display.tag_configure(
            "system", font=(ff, 10, "italic"),
            foreground="#9ca3af", justify="right", spacing1=6)

        # ─── نوار ورودی ────────────────────────────────────────────
        inp = tk.Frame(self.chat_frame, bg=self.BG)
        inp.pack(side="bottom", fill="x", pady=(8, 0))

        self.input_entry = tk.Entry(
            inp, font=(self.font_family, 12),
            bg="#ffffff", fg=self.TEXT_FG,
            relief="solid", borderwidth=1,
            justify="right",
        )
        self.input_entry.pack(side="left", fill="x", expand=True,
                              ipady=10, padx=(0, 8))
        self.input_entry.bind("<Return>", self.on_send)

        self.send_btn = ttk.Button(inp, text="ارسال و پاسخ  ⏎",
                                   command=self.on_send,
                                   style="Accent.TButton")
        self.send_btn.pack(side="right")

        # پیام خوش‌آمد
        self._add_system_message(
            "به سیستم پرسش و پاسخ خوش آمدید.\n\n"
            "روش کار:\n"
            "  ۱) روی «📂 انتخاب پوشه» بزنید و پوشه‌ی فایل‌های .jsonl "
            "را انتخاب کنید.\n"
            "  ۲) روی «🔍 افزودن به ایندکس» بزنید تا ایندکس ساخته شود.\n"
            "  ۳) سؤالتان را در کادر پایین بنویسید و Enter بزنید.\n\n"
            "💡 نکته: با چک‌باکس «🤖 تولید پاسخ با LLM» می‌توانید "
            "پاسخ‌گویی مدل را خاموش کنید و فقط متن‌های بازیابی‌شده "
            "را ببینید."
        )

    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg="#eef2ff", height=28)
        bar.pack(side="bottom", fill="x")
        bar.pack_propagate(False)

        self.status_label = tk.Label(
            bar, text="آماده",
            bg="#eef2ff", fg=self.ACCENT_DARK,
            font=(self.font_family, 10),
            anchor="e", padx=14,
        )
        self.status_label.pack(side="right", fill="both", expand=True)

        self.ollama_status = tk.Label(
            bar, text="Ollama: بررسی…",
            bg="#eef2ff", fg="#6b7280",
            font=(self.font_family, 9), padx=14,
        )
        self.ollama_status.pack(side="left")

        self.root.after(500, self._check_ollama_async)

    def _check_ollama_async(self):
        def worker():
            ok = check_ollama(self.settings.ollama_url)
            self.ui_queue.put(("ollama", ok))
        threading.Thread(target=worker, daemon=True).start()

    # ─── پیام‌های چت ───────────────────────────────────────────────
    def _add_system_message(self, text: str):
        self.chat_display.configure(state="normal")
        self.chat_display.insert("end", text + "\n", "system")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    def _add_message(self, who: str, text: str,
                     sources: list[dict] | None = None,
                     use_llm: bool = True):
        self.chat_display.configure(state="normal")

        if who == "user":
            self.chat_display.insert("end", "👤  شما\n", "user_label")
            self.chat_display.insert("end", text + "\n", "user_text")

        else:  # bot
            # ─── بخش ۱: متن‌های بازیابی‌شده ────────────────────────
            if sources:
                n = len(sources)
                header = f"📚  متن‌های بازیابی‌شده ({n} مورد)\n"
                self.chat_display.insert("end", header, "src_section")

                for i, s in enumerate(sources, 1):
                    hdr = (f"━━━ [{i}] {s['book']} "
                           f"— ص {s['page']} ━━━\n")
                    self.chat_display.insert("end", hdr, "src_header")

                    body = s.get("text", "").strip()
                    if body:
                        self.chat_display.insert("end", body + "\n",
                                                 "src_text")

            # ─── بخش ۲: پاسخ LLM ───────────────────────────────────
            if use_llm and text:
                self.chat_display.insert(
                    "end", "\n🤖  پاسخ مدل\n", "bot_label")
                self.chat_display.insert("end", text + "\n", "bot_text")
            elif not use_llm:
                self.chat_display.insert(
                    "end",
                    "\nℹ  (LLM خاموش است — فقط متن‌های بازیابی‌شده "
                    "نمایش داده شد.)\n",
                    "info")

        self.chat_display.insert("end", "\n")
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

    # ─── بازیابی جلسه ───────────────────────────────────────────────
    def _try_restore_session(self):
        ok, err = self.engine.restore_session()
        if not ok:
            self._add_system_message(
                "ℹ  جلسه‌ی قبلی قابل بازیابی نبود:\n"
                f"    {err}\n\n"
                "لطفاً پوشه‌ی فایل‌های .jsonl را انتخاب کنید."
            )
            self._set_status("آماده. ابتدا پوشه را انتخاب کنید.")
            return

        self._refresh_book_list()
        self.stat_chunks.configure(
            text=f"تکه‌ها: {self.engine.total_chunks:,}")
        self.send_btn.configure(state="normal")
        self._set_status("✅ جلسه‌ی قبلی بازیابی شد. آماده‌ی پرسش.")
        self._add_system_message(
            f"✅ جلسه‌ی قبلی بازیابی شد\n"
            f"   • {len(self.engine.books)} کتاب\n"
            f"   • {self.engine.total_chunks:,} تکه\n"
            f"   • ذخیره‌شده در: {self.engine.session.indexed_at}\n"
            f"   • مدل: {self.settings.llm_model}"
        )
        self.input_entry.focus_set()

    def _refresh_book_list(self):
        self.books_list.delete(0, "end")
        for b in self.engine.books:
            self.books_list.insert(
                "end", f"📕 {b.name}  ({b.useful_pages} ص)")

        total_pages = sum(b.useful_pages for b in self.engine.books)
        total_chars = sum(b.chars for b in self.engine.books)
        self.stat_books.configure(text=f"کتاب‌ها: {len(self.engine.books)}")
        self.stat_pages.configure(text=f"صفحات مفید: {total_pages:,}")
        self.stat_chars.configure(text=f"کاراکترها: {total_chars:,}")

    # ─── انتخاب پوشه ────────────────────────────────────────────────
    def pick_folder(self):
        folder = filedialog.askdirectory(
            title="پوشه‌ای که فایل‌های .jsonl در آن است را انتخاب کنید")
        if not folder:
            return
        folder_path = Path(folder)
        self._set_status("در حال اسکن پوشه…")
        self.index_btn.configure(state="disabled")
        self.rebuild_btn.configure(state="disabled")

        def worker():
            try:
                books = self.engine.scan_folder(
                    folder_path, self.settings.recursive_scan)
                self.ui_queue.put(("scan_done", books))
            except Exception as e:
                self.ui_queue.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_scan_done(self, books: list[Book]):
        self._refresh_book_list()
        self.index_btn.configure(state="normal")
        self.rebuild_btn.configure(state="normal")

        if not books:
            self._set_status("هیچ فایل .jsonl معتبری پیدا نشد.")
            messagebox.showwarning(
                "بدون نتیجه",
                "هیچ فایل .jsonl معتبری پیدا نشد.\n"
                "مطمئن شوید فایل‌ها در پوشه انتخاب‌شده هستند.")
            return

        total_pages = sum(b.useful_pages for b in books)
        self._set_status(
            f"{len(books)} کتاب پیدا شد. حالا «افزودن به ایندکس» "
            f"را بزنید.")
        self._add_system_message(
            f"✅ {len(books)} کتاب اسکن شد — {total_pages:,} "
            f"صفحه‌ی مفید.")

    # ─── ایندکس کردن ────────────────────────────────────────────────
    def start_index(self, rebuild: bool = False):
        if not self.engine.books:
            messagebox.showwarning("بدون کتاب",
                                   "ابتدا پوشه‌ی فایل‌ها را انتخاب کنید.")
            return

        if rebuild:
            if not messagebox.askyesno(
                "تأیید بازسازی",
                "همه‌ی ایندکس قبلی پاک می‌شود و از صفر ساخته "
                "می‌شود.\n"
                "این کار ممکن است چند دقیقه طول بکشد.\n\n"
                "ادامه می‌دهید؟"
            ):
                return

        self.index_btn.configure(state="disabled")
        self.rebuild_btn.configure(state="disabled")
        self.send_btn.configure(state="disabled")
        self.progress["value"] = 0
        self.progress_label.configure(text="0%")
        msg = ("♻ بازسازی کامل ایندکس…" if rebuild
               else "🔍 افزودن به ایندکس…")
        self._add_system_message(f"⏳ {msg}")

        def worker():
            try:
                def on_progress(done, total):
                    pct = int(done * 100 / max(total, 1))
                    self.ui_queue.put(
                        ("progress", pct, f"{done:,}/{total:,}"))

                def on_status(msg):
                    self.ui_queue.put(("status", msg))

                self.engine.build_index(
                    rebuild=rebuild,
                    progress_cb=on_progress,
                    status_cb=on_status,
                )
                self.ui_queue.put(("index_done", self.engine.total_chunks))
            except Exception as e:
                self.ui_queue.put(("error", f"خطا در ایندکس: {e}"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_index_done(self, chunks: int):
        self.stat_chunks.configure(text=f"تکه‌ها: {chunks:,}")
        self.progress["value"] = 100
        self.progress_label.configure(text="100%")
        self.index_btn.configure(state="normal")
        self.rebuild_btn.configure(state="normal")
        self.send_btn.configure(state="normal")
        self._set_status("✅ ایندکس آماده است. سؤال خود را بپرسید.")
        self._add_system_message(
            f"✅ ایندکس آماده شد — {chunks:,} تکه در پایگاه داده.\n"
            "این جلسه ذخیره شد؛ دفعه‌ی بعد خودکار بازیابی می‌شود.")
        self.input_entry.focus_set()

    # ─── ارسال سؤال ─────────────────────────────────────────────────
    def on_send(self, event=None):
        q = self.input_entry.get().strip()
        if not q:
            return "break"

        # چک ایندکس
        if not self.engine.retriever:
            messagebox.showinfo("ایندکس آماده نیست",
                                "ابتدا روی «افزودن به ایندکس» بزنید.")
            return "break"

        use_llm = self.use_llm_var.get()
        if use_llm and not self.engine.chain:
            messagebox.showinfo("ایندکس آماده نیست",
                                "ابتدا روی «افزودن به ایندکس» بزنید.")
            return "break"

        self.input_entry.delete(0, "end")
        self._add_message("user", q)
        self.send_btn.configure(state="disabled")
        self.input_entry.configure(state="disabled")
        self._set_status(
            "در حال جستجو…" if not use_llm else "در حال فکر کردن…")

        def worker():
            try:
                answer, sources = self.engine.query(q, use_llm=use_llm)
                self.ui_queue.put(("answer", answer, sources, use_llm))
            except Exception as e:
                self.ui_queue.put(("error", f"خطا در پاسخ: {e}"))

        threading.Thread(target=worker, daemon=True).start()
        return "break"

    def _on_answer(self, answer: str, sources: list[dict], use_llm: bool):
        self._add_message("bot", answer, sources, use_llm=use_llm)
        self.send_btn.configure(state="normal")
        self.input_entry.configure(state="normal")
        self.input_entry.focus_set()
        self._set_status("آماده")

    # ─── تنظیمات ────────────────────────────────────────────────────
    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("تنظیمات")
        win.geometry("580x620")
        win.configure(bg=self.BG)
        win.transient(self.root)
        win.grab_set()

        container = tk.Frame(win, bg=self.BG)
        container.pack(fill="both", expand=True, padx=20, pady=20)

        s = self.settings
        vars_: dict[str, tk.Variable] = {}

        def row(label: str, var: tk.Variable, hint: str = ""):
            f = tk.Frame(container, bg=self.BG)
            f.pack(fill="x", pady=6)
            tk.Label(f, text=label, bg=self.BG,
                     fg=self.TEXT_FG,
                     font=(self.font_family, 11, "bold"),
                     anchor="e", width=22).pack(side="right")
            tk.Entry(f, textvariable=var,
                     font=(self.font_family, 11),
                     justify="right").pack(
                side="right", fill="x", expand=True, padx=(0, 8))
            if hint:
                tk.Label(f, text=hint, bg=self.BG, fg="#9ca3af",
                         font=(self.font_family, 9),
                         anchor="e").pack(side="right")

        vars_["ollama_url"]    = tk.StringVar(value=s.ollama_url)
        row("آدرس Ollama", vars_["ollama_url"],
            "http://localhost:11434")

        vars_["embed_model"]   = tk.StringVar(value=s.embed_model)
        row("مدل Embedding", vars_["embed_model"],
            "nomic-embed-text یا bge-m3")

        vars_["llm_model"]     = tk.StringVar(value=s.llm_model)
        row("مدل پاسخ‌گو", vars_["llm_model"],
            "qwen2.5:7b | deepseek-r1:7b | llama3.1")

        vars_["chunk_size"]    = tk.IntVar(value=s.chunk_size)
        row("اندازه‌ی تکه", vars_["chunk_size"], "پیشنهاد: 300–500")

        vars_["chunk_overlap"] = tk.IntVar(value=s.chunk_overlap)
        row("همپوشانی", vars_["chunk_overlap"], "پیشنهاد: 10–20٪")

        vars_["top_k"]         = tk.IntVar(value=s.top_k)
        row("تعداد نتایج", vars_["top_k"], "3 تا 10")

        vars_["temperature"]   = tk.DoubleVar(value=s.temperature)
        row("دمای مدل", vars_["temperature"], "0.0 تا 0.3")

        vars_["recursive_scan"] = tk.BooleanVar(value=s.recursive_scan)
        tk.Checkbutton(
            container, text="اسکن بازگشتی پوشه‌ها (زیرپوشه‌ها)",
            variable=vars_["recursive_scan"],
            bg=self.BG, font=(self.font_family, 10),
            anchor="e", justify="right",
        ).pack(anchor="e", pady=(10, 4))

        tk.Label(
            container,
            text=("⚠  اگر مدل Embedding را عوض کنید، ایندکس قبلی "
                  "بی‌اعتبار می‌شود و باید بازسازی کنید."),
            bg=self.BG, fg="#b45309",
            font=(self.font_family, 9),
            wraplength=500, justify="right",
        ).pack(anchor="e", pady=(10, 0))

        btns = tk.Frame(container, bg=self.BG)
        btns.pack(fill="x", pady=(20, 0))

        def save_and_close():
            try:
                new = Settings(
                    ollama_url=vars_["ollama_url"].get().strip(),
                    embed_model=vars_["embed_model"].get().strip(),
                    llm_model=vars_["llm_model"].get().strip(),
                    chunk_size=int(vars_["chunk_size"].get()),
                    chunk_overlap=int(vars_["chunk_overlap"].get()),
                    top_k=int(vars_["top_k"].get()),
                    temperature=float(vars_["temperature"].get()),
                    recursive_scan=bool(vars_["recursive_scan"].get()),
                    use_llm=self.use_llm_var.get(),
                )
                if new.chunk_size < 50:
                    raise ValueError(
                        "اندازه‌ی تکه باید حداقل ۵۰ باشد.")
                if new.chunk_overlap >= new.chunk_size:
                    raise ValueError(
                        "همپوشانی باید کمتر از اندازه‌ی تکه باشد.")
                if new.top_k < 1:
                    raise ValueError("تعداد نتایج باید حداقل ۱ باشد.")
            except Exception as e:
                messagebox.showerror("مقدار نامعتبر", str(e), parent=win)
                return

            embed_changed = (new.embed_model != self.settings.embed_model)
            self.settings = new
            self.engine.settings = new
            self.settings.save()
            self._check_ollama_async()
            if embed_changed:
                messagebox.showwarning(
                    "مدل Embedding تغییر کرد",
                    "مدل Embedding عوض شد. لطفاً «بازسازی کامل» "
                    "را بزنید.",
                    parent=win)
            else:
                messagebox.showinfo("ذخیره شد",
                                    "تنظیمات ذخیره شد.", parent=win)
            win.destroy()

        ttk.Button(btns, text="ذخیره", command=save_and_close,
                   style="Accent.TButton").pack(side="left", padx=4)
        ttk.Button(btns, text="انصراف", command=win.destroy) \
            .pack(side="left", padx=4)

    # ─── پاک کردن همه چیز ───────────────────────────────────────────
    def clear_books(self):
        if not messagebox.askyesno(
            "تأیید حذف",
            "لیست کتاب‌ها، ایندکس و تنظیمات جلسه پاک می‌شود.\n"
            "این کار غیرقابل بازگشت است.\n\nادامه؟"
        ):
            return
        try:
            if self.engine.chroma_dir.exists():
                shutil.rmtree(self.engine.chroma_dir)
        except Exception:
            pass
        self.engine.session.clear()

        self.engine.books = []
        self.engine.chain = None
        self.engine.retriever = None
        self.engine.total_chunks = 0
        self.books_list.delete(0, "end")
        for w in (self.stat_books, self.stat_pages,
                  self.stat_chars, self.stat_chunks):
            w.configure(text=w.cget("text").split(":")[0] + ": 0")
        self.progress["value"] = 0
        self.progress_label.configure(text="")
        self._set_status("همه چیز پاک شد.")
        self._add_system_message("🗑  همه چیز پاک شد.")

    # ─── نوار وضعیت ─────────────────────────────────────────────────
    def _set_status(self, msg: str):
        self.status_label.configure(text=msg)

    # ─── حلقه‌ی صف UI ───────────────────────────────────────────────
    def _poll_ui_queue(self):
        try:
            while True:
                item = self.ui_queue.get_nowait()
                kind = item[0]
                if kind == "scan_done":
                    self._on_scan_done(item[1])
                elif kind == "index_done":
                    self._on_index_done(item[1])
                elif kind == "answer":
                    _, ans, srcs, use_llm = item
                    self._on_answer(ans, srcs, use_llm)
                elif kind == "error":
                    self._set_status("⚠ " + item[1])
                    self.index_btn.configure(state="normal")
                    self.rebuild_btn.configure(state="normal")
                    self.send_btn.configure(state="normal")
                    self.input_entry.configure(state="normal")
                    messagebox.showerror("خطا", item[1])
                elif kind == "progress":
                    pct, txt = item[1], item[2]
                    self.progress["value"] = pct
                    self.progress_label.configure(text=f"{pct}%  ({txt})")
                elif kind == "status":
                    self._set_status(item[1])
                elif kind == "ollama":
                    if item[1]:
                        self.ollama_status.configure(
                            text="Ollama: متصل ✅", fg="#047857")
                    else:
                        self.ollama_status.configure(
                            text="Ollama: قطع ❌", fg="#dc2626")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_ui_queue)

    # ─── اجرا ───────────────────────────────────────────────────────
    def run(self):
        self.input_entry.focus_set()
        # اعمال متن دکمه بر اساس تنظیمات اولیه
        if not self.use_llm_var.get():
            self.send_btn.configure(text="جستجو  ⏎")
        self.root.mainloop()


def main():
    app = RAGApp()
    app.run()


if __name__ == "__main__":
    main()