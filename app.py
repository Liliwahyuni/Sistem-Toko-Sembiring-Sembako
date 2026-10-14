"""Sistem Analisis Pola Pembelian Toko Sembiring Sembako - Apriori vs FP-Growth (Streamlit).

Jalankan:  streamlit run app.py
"""
import math
import re
from datetime import date
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import database as db
import mining as mn
import preprocessing as pp
import report as rp
import styles as sty

st.set_page_config(page_title="Analisis Pola Pembelian - Toko Sembiring Sembako", page_icon="🛒", layout="wide")
db.init_db()
sty.inject()

MENU = ["Dashboard", "Kelola Data", "Analisis Algoritma", "Perbandingan dan Rekomendasi",
        "Riwayat Analisis", "Ekspor Laporan"]
SUPPORT_OPT = [0.5, 1, 2, 5, 10, 15, 20]
CONF_OPT = [50, 60, 70, 80, 90]
S = st.session_state
S.setdefault("user", None)
S.setdefault("page", "Dashboard")


# =============================================================== util UI
def topbar(title):
    st.markdown(f'<div class="topbar"><div class="page-title">{title}</div>'
                '<div class="badge">Pemilik Toko</div></div>', unsafe_allow_html=True)
    if S.get("flash"):
        st.success(S.pop("flash"))


def section(t):
    st.markdown(f'<div class="section">{t}</div>', unsafe_allow_html=True)


def tab_buttons(options, key):
    S.setdefault(key, options[0])
    cols = st.columns(len(options) + 3)
    for i, o in enumerate(options):
        if cols[i].button(o, key=f"{key}_{o}", type="primary" if S[key] == o else "secondary", width="stretch"):
            S[key] = o
            st.rerun()
    return S[key]


@st.dialog("Konfirmasi Hapus Data")
def confirm_delete(pesan, aksi, ketik=None):
    st.warning(pesan)
    ok = True
    if ketik:
        ok = st.text_input(f'Ketik "{ketik}" untuk melanjutkan').strip() == ketik
    c1, c2 = st.columns(2)
    if c1.button("Ya, Hapus", key="danger_ok", disabled=not ok, width="stretch"):
        S["flash"] = aksi()
        st.rerun()
    if c2.button("Batal", key="dlg_cancel", width="stretch"):
        st.rerun()


# =============================================================== matriks one hot
@st.cache_data(show_spinner=False)
def onehot_cached(df):
    """One-hot dibuat sekali lalu di-cache, supaya tidak dihitung ulang tiap klik."""
    return pp.one_hot(df)


@st.cache_data(show_spinner=False)
def matriks_urut(df):
    """Matriks One Hot untuk TAMPILAN: baris = nomor transaksi (diurutkan), kolom = produk, nilai 1/0.
    Dibentuk langsung dari data transaksi sehingga nomor transaksi selalu menjadi baris."""
    m = pd.crosstab(df["no_transaksi"], df["nama_produk"]).clip(upper=1).astype(int)
    m = m.loc[sorted(m.index, key=_kunci_alami)]
    m.index.name = "No Transaksi"
    return m


def _kunci_alami(nilai):
    """Kunci urut alami: 'TRX2' < 'TRX10', 2 < 10. Pola re.split selalu teks, angka, teks, ... sehingga aman dibandingkan."""
    return tuple(int(t) if i % 2 else t.lower() for i, t in enumerate(re.split(r"(\d+)", str(nilai))))


def render_onehot(basket, key):
    """Tampilkan matriks One Hot Encoding (baris = transaksi, kolom = produk, nilai 1/0),
    diurutkan berdasarkan nomor transaksi."""
    b = basket.astype(int)  # True/False -> 1/0
    b = b.loc[sorted(b.index, key=_kunci_alami)]  # urut sesuai nomor transaksi
    b.index.name = "No Transaksi"
    n_trx, n_prod = b.shape
    total1 = int(b.values.sum())
    kepadatan = total1 / (n_trx * n_prod) * 100 if n_trx and n_prod else 0

    c = st.columns(4)
    with c[0]: sty.card("Jumlah Transaksi (baris)", f"{n_trx:,}".replace(",", "."), mini=True)
    with c[1]: sty.card("Jumlah Produk (kolom)", n_prod, mini=True)
    with c[2]: sty.card("Jumlah Nilai 1", f"{total1:,}".replace(",", "."), mini=True)
    with c[3]: sty.card("Kepadatan Matriks", f"{kepadatan:.2f}%", mini=True)
    st.caption("1 = produk dibeli pada transaksi tersebut, 0 = tidak dibeli. "
               "Matriks ini adalah data masukan Apriori dan FP-Growth.")

    k1, k2, k3 = st.columns([3, 1, 1])
    kolom = k1.multiselect("Pilih kolom produk (kosong = 15 produk pertama)", list(b.columns),
                           key=f"oh_col_{key}")
    awal = k2.number_input("Mulai dari transaksi ke-", min_value=1, max_value=max(1, n_trx),
                           value=1, step=20, key=f"oh_start_{key}")
    baris = k3.number_input("Jumlah baris", min_value=1, max_value=max(1, min(n_trx, 500)),
                            value=min(20, n_trx), step=5, key=f"oh_row_{key}")

    i0 = int(awal) - 1
    view = b[kolom or list(b.columns[:15])].iloc[i0:i0 + int(baris)]
    st.caption(f"Menampilkan transaksi urutan ke-{i0 + 1} s.d. ke-{min(i0 + int(baris), n_trx)} "
               f"dari {n_trx}, diurutkan berdasarkan nomor transaksi.")
    stl = view.style
    warnai = getattr(stl, "map", None) or stl.applymap  # pandas baru: map, lama: applymap
    st.dataframe(warnai(lambda v: "background-color:#2f6f5e;color:white;font-weight:600"
                        if v == 1 else "color:#8a8f98"), width="stretch")

    with st.expander("Jumlah transaksi per produk (jumlah angka 1 tiap kolom)"):
        rk = pd.DataFrame({"Produk": b.columns, "Jumlah Transaksi": b.sum().values})
        rk["Support (%)"] = (rk["Jumlah Transaksi"] / n_trx * 100).round(2)
        st.dataframe(rk.sort_values("Jumlah Transaksi", ascending=False), hide_index=True, width="stretch")

    st.download_button("Unduh Matriks (CSV)", b.to_csv().encode("utf-8"),
                       file_name="matriks_one_hot.csv", mime="text/csv", key=f"oh_dl_{key}")


# =============================================================== login
def page_login():
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        st.markdown('<div class="login-wrap"><br><br><h1>Sistem Analisis Pola Pembelian<br>Toko Sembiring Sembako</h1>'
                    '<div class="login-sub">Silakan Login untuk Masuk ke Sistem</div></div>', unsafe_allow_html=True)
        with st.container(border=True):
            with st.form("login"):
                u = st.text_input("Username", placeholder="Masukkan Username", max_chars=10)
                p = st.text_input("Password", placeholder="Masukkan Password", type="password")
                go_ = st.form_submit_button("Login", type="primary", width="stretch")
            if go_:
                user = db.verify_user(u, p)
                if user:
                    S["user"], S["page"] = user, "Dashboard"
                    st.rerun()
                else:
                    st.error("Username atau password salah. Silakan coba lagi.")


# =============================================================== sidebar
def sidebar():
    with st.sidebar:
        st.markdown('<div class="brand">Sistem Analisis Pola Pembelian<br>Toko Sembiring Sembako'
                    '<small>APRIORI · FP-GROWTH</small></div>', unsafe_allow_html=True)
        for m in MENU:
            if st.button(m, key=f"nav_{m}", type="primary" if S["page"] == m else "secondary", width="stretch"):
                S["page"] = m
                st.rerun()
        st.markdown("<br>" * 4, unsafe_allow_html=True)
        if st.button("Logout", key="logout", width="stretch"):
            S["user"] = None
            st.rerun()


# =============================================================== dashboard
def page_dashboard():
    topbar("Halaman Dashboard")
    st.markdown('<div class="welcome">SELAMAT DATANG!!</div>', unsafe_allow_html=True)
    d = db.dashboard_stats()
    last = d["last"]
    _, c1, c2, _ = st.columns([.3, 1, 1, .3])
    with c1:
        sty.card("Total Transaksi", f'{d["transaksi"]:,}'.replace(",", "."), "Data transaksi yang tersimpan")
    with c2:
        sty.card("Jumlah Analisis", d["analisis"], "Total analisis yang dilakukan")
    st.write("")
    _, c3, c4, _ = st.columns([.3, 1, 1, .3])
    with c3:
        if last:
            sty.card("Analisis Terakhir", last["tanggal_analisis"].replace("T", " "),
                     f'{last["nama_algoritma"]} · Min Support {last["minimum_support"]:g}%', small=True)
        else:
            sty.card("Analisis Terakhir", "-")
    with c4:
        sty.card("Jumlah Frequent Analisis", last["jumlah_frequent_itemset"] if last else "-",
                 f'Hasil {last["nama_algoritma"]}' if last else "")


# =============================================================== kelola data
def page_kelola():
    topbar("Halaman Kelola Data")
    st.markdown("<h3 style='text-align:center'>Unggah Data Transaksi</h3>", unsafe_allow_html=True)
    st.markdown('<div class="subtle">Unggah file excel transaksi penjualan untuk dikelola sistem</div>',
                unsafe_allow_html=True)
    S.setdefault("up_n", 0)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        f = st.file_uploader("Pilih File  ·  Format file .xlsx", type=["xlsx"], key=f"up_{S['up_n']}")

    if f is not None:
        try:
            raw = pp.read_excel(f)
        except Exception as e:  # format / kolom tidak sesuai
            st.error(f"File tidak dapat dibaca: {e}")
            raw = None
        if raw is not None:
            section("Pratinjau Data")
            prev = raw.head(15).rename(columns=pp.TAMPIL)
            st.dataframe(prev, hide_index=True, width="stretch")
            st.caption(f"Menampilkan 15 dari {len(raw):,} baris data mentah.".replace(",", "."))
            b1, b2, _ = st.columns([1, 1, 4])
            if b1.button("Unggah Data", type="primary", width="stretch"):
                with st.spinner("Memproses data (preprocessing)…"):
                    clean, stats = pp.preprocess(raw)
                    ada = db.existing_transaction_numbers()
                    baru = clean[~clean["no_transaksi"].isin(ada)]
                    stats["dilewati"] = len(clean) - len(baru)
                    if len(baru):
                        db.insert_transactions(baru, S["user"]["user_id"])
                S["upload_stats"] = stats
                S["up_n"] += 1  # kosongkan uploader
                st.rerun()
            if b2.button("Batal", width="stretch"):
                S["up_n"] += 1
                S.pop("upload_stats", None)
                st.rerun()

    st_ = S.get("upload_stats")
    if st_:
        st.markdown('<div class="success-box">✔ Data Berhasil diunggah dan Diprocessing</div>', unsafe_allow_html=True)
        c = st.columns(4)
        with c[0]: sty.card("Total Data Awal", f'{st_["total_awal"]:,}'.replace(",", "."), mini=True)
        with c[1]: sty.card("Total Data Akhir", f'{st_["total_akhir"]:,}'.replace(",", "."), mini=True)
        with c[2]: sty.card("Total Produk", st_["total_produk"], mini=True)
        with c[3]: sty.card("One Hot Encoding", f'{st_["onehot_shape"][0]} × {st_["onehot_shape"][1]}', mini=True)
        with st.expander("Rincian proses preprocessing"):
            st.dataframe(pd.DataFrame({"Tahap pembersihan": list(st_["log"]), "Baris dibuang": list(st_["log"].values())}),
                         hide_index=True, width="stretch")
            if st_.get("dilewati"):
                st.info(f'{st_["dilewati"]} baris dilewati karena nomor transaksinya sudah ada di database.')
            st.caption("Matriks one-hot hanya dibuat sementara di memori saat analisis (tidak disimpan sebagai tabel).")

    # ----- data tersimpan + hapus data
    section("Data Transaksi Tersimpan")
    s = db.transaction_stats()
    if not s["b"]:
        st.info("Belum ada data transaksi. Unggah file Excel terlebih dahulu.")
        return
    st.caption(f'{s["t"]} transaksi · {s["p"]} produk · {s["b"]} baris · periode {s["a"]} s.d. {s["z"]}')
    st.dataframe(db.get_transactions().rename(columns=pp.TAMPIL), hide_index=True, width="stretch", height=300)

    # ----- matriks one hot encoding
    section("Matriks One Hot Encoding")
    with st.expander("Lihat matriks One Hot Encoding dari data tersimpan"):
        render_onehot(matriks_urut(db.get_transactions()), "kelola")

    section("Hapus Data Transaksi")
    with st.container(border=True):
        mode = st.radio("Pilih cara penghapusan", ["Per nomor transaksi", "Per rentang tanggal", "Semua data transaksi"],
                        horizontal=True)
        df = db.get_transactions()
        if mode == "Per nomor transaksi":
            pilih = st.multiselect("Pilih No Transaksi yang akan dihapus", sorted(df["no_transaksi"].unique()))
            if st.button("Hapus Data", key="danger_no", disabled=not pilih):
                confirm_delete(f"{len(pilih)} transaksi akan dihapus permanen.",
                               lambda: f"{db.delete_transactions_by_no(pilih)} baris transaksi berhasil dihapus.")
        elif mode == "Per rentang tanggal":
            a, z = date.fromisoformat(s["a"]), date.fromisoformat(s["z"])
            rng = st.date_input("Rentang tanggal", value=(a, z), min_value=a, max_value=z)
            if st.button("Hapus Data", key="danger_rng", disabled=len(rng) != 2):
                confirm_delete(f"Semua transaksi tanggal {rng[0]} s.d. {rng[1]} akan dihapus permanen.",
                               lambda: f"{db.delete_transactions_by_period(rng[0].isoformat(), rng[1].isoformat())} "
                                       "baris transaksi berhasil dihapus.")
        else:
            st.caption("Hasil analisis & riwayat tidak ikut terhapus (hapus di halaman Riwayat Analisis).")
            if st.button("Hapus Semua Data", key="danger_all"):
                confirm_delete("SELURUH data transaksi akan dihapus permanen dan tidak dapat dikembalikan.",
                               lambda: f"{db.delete_all_transactions()} baris transaksi berhasil dihapus.", ketik="HAPUS")


# =============================================================== conviction
def _frac(v):
    """Normalisasi nilai ke pecahan 0-1 (nilai > 1 dianggap persen)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    v = float(v)
    return v / 100 if v > 1 else v


def conviction(support_b, confidence):
    """Conviction (A -> B) = (1 - Support(B)) / (1 - Confidence(A,B)).

    - 1        : A dan B independen
    - > 1      : aturan semakin akurat / jarang meleset
    - inf      : confidence = 1 (aturan tidak pernah meleset)
    Support(B) dan Confidence boleh berupa pecahan (0-1) atau persen (0-100).
    """
    sb, cf = _frac(support_b), _frac(confidence)
    if sb is None or cf is None:
        return None
    if cf >= 1:
        return math.inf
    return (1 - sb) / (1 - cf)


def _ambil(rule, *names):
    for n in names:
        if n in rule and rule[n] is not None:
            return rule[n]
    return None


def tambah_conviction(df, rules, itemsets):
    """Tambahkan kolom Conviction ke tabel aturan hasil rp.rules_df().

    Support(B) dicari dari aturan itu sendiri; bila tidak ada, dicari dari
    daftar frequent itemset berdasarkan itemset konsekuen.
    """
    if df is None or "Conviction" in df.columns or len(df) != len(rules):
        return df
    sup_item = {}
    for it in itemsets or []:
        items = _ambil(it, "itemset", "items")
        sup = _ambil(it, "support")
        if items is not None and sup is not None:
            sup_item[frozenset(items if not isinstance(items, str) else [items])] = sup
    nilai = []
    for r in rules:
        sb = _ambil(r, "support_consequent", "consequent_support", "support_konsekuen", "sup_b")
        if sb is None:
            cons = _ambil(r, "consequents", "consequent", "konsekuen")
            if cons is not None:
                cons = frozenset([cons] if isinstance(cons, str) else cons)
                sb = sup_item.get(cons)
        cf = _ambil(r, "confidence", "conf")
        c = conviction(sb, cf)
        if c is None:
            nilai.append("-")
        elif math.isinf(c):
            nilai.append("∞")
        else:
            nilai.append(round(c, 4))
    df = df.copy()
    df["Conviction"] = nilai
    return df


# =============================================================== hasil analisis (dipakai Analisis & Riwayat)
_CSS_TBL = (
    "<style>"
    ".tbl-fit{width:100%;border-collapse:collapse;font-size:13px;}"
    ".tbl-fit th,.tbl-fit td{border:1px solid rgba(128,128,128,.35);padding:5px 7px;text-align:left;"
    "vertical-align:top;overflow-wrap:anywhere;word-break:break-word;}"
    ".tbl-fit th{background:rgba(128,128,128,.18);font-weight:600;}"
    ".tbl-fit tbody tr:nth-child(even) td{background:rgba(128,128,128,.07);}"
    "</style>"
)


def tabel_html(df):
    """Tabel statis: SEMUA baris tampil, teks panjang dibungkus ke baris berikutnya,
    sehingga tidak perlu scroll ke bawah maupun ke samping."""
    kepala = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    isi = "".join("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
                  for row in df.itertuples(index=False, name=None))
    st.markdown(f'{_CSS_TBL}<table class="tbl-fit"><thead><tr>{kepala}</tr></thead>'
                f'<tbody>{isi}</tbody></table>', unsafe_allow_html=True)


def tabel_penuh(df):
    """Tampilkan SELURUH baris tabel tanpa scroll vertikal; lebar mengikuti layar (tanpa scroll samping)."""
    st.dataframe(df, hide_index=True, width="stretch", height=int(35 * (len(df) + 1) + 3))


_CSS_RINGKAS = (
    "<style>"
    ".sum-row{display:flex;gap:8px;margin:0 0 4px 0;}"
    ".sum-box{flex:1;display:flex;justify-content:space-between;align-items:center;"
    "border:1px solid rgba(128,128,128,.35);border-radius:8px;padding:3px 12px;font-size:13px;}"
    ".sum-box b{font-size:15px;}"
    "</style>"
)


def ringkasan_kecil(a):
    """Tiga ringkasan (frequent, aturan, waktu) dalam satu baris kecil agar tabel hasil lebih lapang."""
    item = [("Jumlah Frequent", a["jumlah_frequent_itemset"]),
            ("Jumlah Aturan", a["jumlah_aturan_asosiasi"]),
            ("Waktu Eksekusi", f'{a["waktu_eksekusi"]:.4f} detik')]
    kotak = "".join(f'<div class="sum-box"><span>{escape(str(l))}</span><b>{escape(str(v))}</b></div>'
                    for l, v in item)
    st.markdown(f'{_CSS_RINGKAS}<div class="sum-row">{kotak}</div>', unsafe_allow_html=True)


def render_result(a: dict, key: str):
    ringkasan_kecil(a)
    label = f'{a["nama_algoritma"]} {a["minimum_support"]:g}%'
    kiri, kanan = st.columns([1, 1.8])
    with kiri:
        section(f"Frequent {label}")
        tabel_html(rp.itemset_df(a["freq"]["itemsets"]).drop(columns="Jumlah Trx"))
    with kanan:
        section(f"Aturan {label}")
        if a["rules"]:
            tabel_html(tambah_conviction(rp.rules_df(a["rules"]), a["rules"], a["freq"]["itemsets"]))
            st.caption("Conviction = (1 − Support B) / (1 − Confidence). Nilai 1 = independen, "
                       "> 1 = aturan jarang meleset, ∞ = confidence 100%.")
        else:
            st.info("Tidak ada aturan yang memenuhi min confidence dan lift > 1.")
    fp = a["freq"].get("fp")
    if a["nama_algoritma"] == "FP-Growth" and fp:
        render_fp(fp, key)


# =============================================================== grup jalur FP-Tree
def _susun_node(tree):
    """Ratakan FP-Tree menjadi daftar node (urutan DFS), lengkap dengan jalur dari ROOT."""
    nodes = []

    def dfs(n, prefix, parent):
        for c in n.get("children", []):
            path = prefix + [c["item"]]
            idx = len(nodes)
            nodes.append({"idx": idx, "item": c["item"], "count": c["count"],
                          "id": c.get("id", c.get("node_id")), "path": path,
                          "parent": parent, "children": []})
            if parent is not None:
                nodes[parent]["children"].append(idx)
            dfs(c, path, idx)

    dfs(tree, [], None)
    return nodes


def _turunan(nodes, idx):
    """Node ini beserta seluruh keturunannya."""
    hasil = [idx]
    for c in nodes[idx]["children"]:
        hasil += _turunan(nodes, c)
    return hasil


def _hitung_grup(nodes, df):
    """Tempatkan tiap transaksi pada node tempat jalurnya berhenti.

    Satu transaksi disisipkan ke FP-Tree sebagai himpunan item frequent yang terurut, sehingga
    himpunan itemnya persis sama dengan himpunan item pada jalur node tempat ia berhenti.
    Jumlah transaksi yang melewati sebuah node = transaksi yang berhenti di node itu + di keturunannya.
    """
    item_pohon = {nd["item"] for nd in nodes}
    kunci = {frozenset(nd["path"]): nd["idx"] for nd in nodes}
    akhir = {nd["idx"]: [] for nd in nodes}
    tak_cocok = 0
    for trx, grp in df.groupby("no_transaksi")["nama_produk"]:
        s = frozenset(grp) & item_pohon
        if not s:
            continue
        i = kunci.get(s)
        if i is None:
            tak_cocok += 1
        else:
            akhir[i].append(trx)
    lewat = {}
    for nd in reversed(nodes):  # anak selalu berada setelah induknya pada urutan DFS
        lewat[nd["idx"]] = len(akhir[nd["idx"]]) + sum(lewat[c] for c in nd["children"])
    return akhir, lewat, tak_cocok


def _nid(nd):
    if nd["id"] is None:
        return ""
    s = str(nd["id"])
    return s if s.upper().startswith("N") else f"N{s}"


def _label_node(nd):
    n = _nid(nd)
    return f'{" → ".join(nd["path"])} ({nd["count"]}{", " + n if n else ""})'


def render_grup_jalur(fp: dict, key: str):
    """Tampilkan transaksi penyusun tiap jalur FP-Tree (mis. Aqua → Plastik Cendana = 84 transaksi)."""
    df = db.get_transactions()
    nodes = _susun_node(fp["tree"])
    if df.empty or not nodes:
        st.info("Data transaksi atau FP-Tree kosong.")
        return
    akhir, lewat, tak_cocok = _hitung_grup(nodes, df)
    if tak_cocok or any(lewat[nd["idx"]] != nd["count"] for nd in nodes):
        st.warning("Daftar transaksi tidak sepenuhnya cocok dengan FP-Tree tersimpan (data transaksi mungkin "
                   "berubah setelah analisis dijalankan). Jalankan ulang analisis agar angkanya sama.")

    akar = [nd["idx"] for nd in nodes if nd["parent"] is None]
    c1, c2 = st.columns(2)
    cab = c1.selectbox("Cabang dari ROOT", akar, key=f"gj_cab_{key}",
                       format_func=lambda i: f'{nodes[i]["item"]} ({nodes[i]["count"]})')
    sel = c2.selectbox("Jalur", _turunan(nodes, cab), key=f"gj_jalur_{key}_{cab}",
                       format_func=lambda i: _label_node(nodes[i]))
    nd = nodes[sel]

    item = [("Jalur", " → ".join(nd["path"])), ("Count di pohon", nd["count"]),
            ("Transaksi melewati jalur", lewat[sel]), ("Berhenti di node ini", len(akhir[sel]))]
    kotak = "".join(f'<div class="sum-box"><span>{escape(str(l))}</span><b>{escape(str(v))}</b></div>'
                    for l, v in item)
    st.markdown(f'{_CSS_RINGKAS}<div class="sum-row">{kotak}</div>', unsafe_allow_html=True)

    mode = st.radio("Transaksi yang ditampilkan",
                    ["Semua yang melewati jalur", "Hanya yang berhenti di node ini"],
                    horizontal=True, key=f"gj_mode_{key}")
    target = _turunan(nodes, sel) if mode.startswith("Semua") else [sel]
    baris = []
    for j in target:
        lanjut = nodes[j]["path"][len(nd["path"]):]
        for trx in akhir[j]:
            baris.append({"No Transaksi": trx,
                          "Item transaksi (urut FP-Tree)": " → ".join(nodes[j]["path"]),
                          "Lanjut setelah jalur": " → ".join(lanjut) if lanjut else "(berhenti di sini)"})
    baris.sort(key=lambda r: _kunci_alami(r["No Transaksi"]))
    tdf = pd.DataFrame(baris, columns=["No Transaksi", "Item transaksi (urut FP-Tree)", "Lanjut setelah jalur"])
    tdf.insert(0, "No", range(1, len(tdf) + 1))

    pecah = []
    for c in nd["children"]:
        n = _nid(nodes[c])
        pecah.append({"Lanjut ke": nodes[c]["item"] + (f" ({n})" if n else ""), "Jumlah transaksi": lewat[c]})
    pecah.append({"Lanjut ke": "(berhenti di node ini)", "Jumlah transaksi": len(akhir[sel])})
    pecah.append({"Lanjut ke": "Total", "Jumlah transaksi": lewat[sel]})

    kiri, kanan = st.columns([1, 2.2])
    with kiri:
        section("Pecahan count")
        tabel_html(pd.DataFrame(pecah))
    with kanan:
        section(f'Transaksi penyusun jalur ({len(tdf)})')
        if len(tdf) <= 120:
            tabel_penuh(tdf)
        else:
            st.dataframe(tdf, hide_index=True, width="stretch", height=600)
        st.caption("Hanya item frequent (yang masuk FP-Tree) yang ditampilkan.")
        st.download_button("Unduh daftar transaksi (CSV)", tdf.to_csv(index=False).encode("utf-8"),
                           file_name="grup_jalur_transaksi.csv", mime="text/csv", key=f"gj_dl_{key}_{sel}_{mode[:5]}")

    with st.expander("Daftar seluruh jalur FP-Tree"):
        tabel_penuh(pd.DataFrame([{"Cabang": n["path"][0], "Jalur": " → ".join(n["path"]), "Count": n["count"],
                                   "Berhenti di node": len(akhir[n["idx"]])} for n in nodes]))


def render_fp(fp: dict, key: str):
    section("Hasil FP-Tree dan Conditional Pattern Base (CPB)")
    st.caption(f'{fp["n_transaksi"]} transaksi · min count = {fp["min_count"]} transaksi · '
               f'{len(fp["header"])} item frequent · {mn.count_nodes(fp["tree"]) - 1} node pada FP-Tree')
    tab = tab_buttons(["Header Table", "FP-Tree", "Grup Jalur", "Conditional Pattern Base"], f"fp_tab_{key}")
    header, cpb_df = rp.fp_frames(fp)

    if tab == "Header Table":
        st.markdown('<div class="small-note">Item frequent diurutkan dari support tertinggi — '
                    'dasar pengurutan transaksi saat membangun FP-Tree.</div>', unsafe_allow_html=True)
        tabel_penuh(header)

    elif tab == "FP-Tree":
        cabang = [c["item"] for c in fp["tree"]["children"]]
        pilihan = ["Seluruh pohon (ROOT)"] + [f'{c["item"]} ({c["count"]})' for c in fp["tree"]["children"]]
        pil = st.selectbox("Tampilkan cabang dari ROOT", pilihan, key=f"br_{key}")
        idx = pilihan.index(pil)
        arah = st.radio("Arah pohon", ["Kiri → Kanan", "Atas → Bawah"], horizontal=True, key=f"dir_{key}")
        n = mn.count_nodes(fp["tree"])
        if idx == 0 and n > 250:
            st.warning("Pohon sangat besar; tampilkan per cabang agar mudah dibaca.")
        else:
            st.graphviz_chart(mn.tree_to_dot(fp["tree"], None if idx == 0 else cabang[idx - 1],
                                             "LR" if arah.startswith("Kiri") else "TB"))
        with st.expander("Tampilan teks FP-Tree"):
            st.code(mn.tree_to_text(fp["tree"]), language=None)

    elif tab == "Grup Jalur":
        render_grup_jalur(fp, key)

    else:
        items = list(fp["cpb"].keys())  # dimulai dari item terakhir header table
        it = st.selectbox("Pilih item (urut dari item terakhir header table)", items, key=f"cpb_{key}")
        d = fp["cpb"][it]
        x1, x2 = st.columns(2)
        with x1: sty.card("Node pada FP-Tree", d["jumlah_node"], mini=True)
        with x2: sty.card(f"Count {it}", d["count_item"], mini=True)
        section(f"Conditional Pattern Base — {it}")
        if d["paths"]:
            tabel_penuh(pd.DataFrame([{"No": i, f"Prefix path sebelum {it}": " → ".join(p["path"]),
                                       f"Count node {it}": p["count"]} for i, p in enumerate(d["paths"], 1)]))
        else:
            st.info(f"CPB {it}: tidak ada prefix path karena semua node {it} langsung berada di bawah ROOT.")
        if d["conditional"]:
            section("Item dalam CPB dan hasil filter min support (Conditional FP-Tree)")
            tabel_penuh(pd.DataFrame([{"Item": c["item"], "Jumlah": c["count"],
                                       "Lolos min support": "Ya" if c["lolos"] else "Tidak"}
                                      for c in d["conditional"]]))
        section(f"Frequent itemset berakhiran {it}")
        tabel_penuh(pd.DataFrame([{"Itemset": ", ".join(p["itemset"]), "Jumlah Trx": p["count"],
                                   "Support": rp.pct(p["support"])} for p in fp["patterns"][it]]))
        with st.expander("Ringkasan seluruh CPB (semua item)"):
            tabel_penuh(cpb_df)


# =============================================================== analisis algoritma
def _simpan_perbandingan(code, sup_pct, an):
    """Simpan perbandingan + rekomendasi setelah KEDUA algoritma selesai (parameter sama)."""
    ap, fg = an["Apriori"], an["FP-Growth"]
    ta, tf = ap["waktu_eksekusi"], fg["waktu_eksekusi"]
    cepat = "Apriori" if ta <= tf else "FP-Growth"
    sama = (ap["jumlah_frequent_itemset"] == fg["jumlah_frequent_itemset"]
            and ap["jumlah_aturan_asosiasi"] == fg["jumlah_aturan_asosiasi"])
    db.save_comparison_result(
        an[cepat]["analysis_id"], code, "Perbandingan",
        f"Perbandingan Apriori vs FP-Growth (min support {sup_pct:g}%)",
        f"Apriori: {ap['jumlah_frequent_itemset']} frequent, {ap['jumlah_aturan_asosiasi']} aturan, {ta:.4f} detik. "
        f"FP-Growth: {fg['jumlah_frequent_itemset']} frequent, {fg['jumlah_aturan_asosiasi']} aturan, {tf:.4f} detik. "
        f"Algoritma tercepat: {cepat}. " + ("Kedua algoritma menghasilkan frequent itemset dan aturan yang sama."
                                           if sama else "Jumlah hasil kedua algoritma berbeda."), cepat)
    for rec in mn.make_recommendations(fg["rules"]):
        db.save_comparison_result(fg["analysis_id"], code, "Rekomendasi", rec["judul"], rec["deskripsi"])


def jalankan_satu(name, sup_pct, conf_pct):
    """Jalankan SATU algoritma (Apriori atau FP-Growth)."""
    df = db.get_transactions()
    if df.empty:
        st.error("Belum ada data transaksi. Unggah data terlebih dahulu di halaman Kelola Data.")
        return
    # pakai kode yang sama bila algoritma lainnya sudah dijalankan dengan parameter identik
    code = S.get("current_code")
    an = db.get_analysis_by_code(code) if code else {}
    cocok = bool(an) and name not in an and all(
        a["minimum_support"] == sup_pct and a["minimum_confidence"] == conf_pct for a in an.values())
    if not cocok:
        code = mn.new_code()

    with st.spinner(f"Menjalankan {name}…"):
        basket = onehot_cached(df)  # one-hot dibuat di luar pengukuran waktu algoritma
        r = mn.run_algorithm(name, basket, sup_pct / 100, conf_pct / 100)
        payload = {"itemsets": r["frequent"]}
        if name == "FP-Growth":
            payload["fp"] = mn.build_fp_structures(basket, sup_pct / 100)
        aid = db.save_analysis(code, name, sup_pct, conf_pct, mn.MIN_LIFT, len(r["frequent"]),
                               len(r["rules"]), r["seconds"], payload, r["rules"])

    an = db.get_analysis_by_code(code)
    pesan = f"{name} selesai dijalankan ({r['seconds']:.4f} detik)."
    if len(an) == 2 and not db.get_results(code, "Perbandingan"):
        _simpan_perbandingan(code, sup_pct, an)
        pesan += " Kedua algoritma sudah selesai, perbandingan & rekomendasi tersimpan."
    S["current_code"] = code
    S[f"aid_{name}"] = aid  # hasil yang ditampilkan di bawah card algoritma ini
    S["flash"] = pesan
    st.rerun()


def jalankan_analisis(sup_pct, conf_pct):
    """Jalankan Apriori dan FP-Growth sekaligus (sub halaman 'Jalankan Kedua Algoritma')."""
    df = db.get_transactions()
    if df.empty:
        st.error("Belum ada data transaksi. Unggah data terlebih dahulu di halaman Kelola Data.")
        return
    with st.spinner("Menjalankan Apriori dan FP-Growth…"):
        basket = onehot_cached(df)
        code, ids, res = mn.new_code(), {}, {}
        for name in ("Apriori", "FP-Growth"):
            r = mn.run_algorithm(name, basket, sup_pct / 100, conf_pct / 100)
            payload = {"itemsets": r["frequent"]}
            if name == "FP-Growth":
                payload["fp"] = mn.build_fp_structures(basket, sup_pct / 100)
            ids[name] = db.save_analysis(code, name, sup_pct, conf_pct, mn.MIN_LIFT, len(r["frequent"]),
                                         len(r["rules"]), r["seconds"], payload, r["rules"])
            res[name] = r
        # perbandingan + rekomendasi disimpan ke comparison_recommendations
        ta, tf = res["Apriori"]["seconds"], res["FP-Growth"]["seconds"]
        cepat = "Apriori" if ta <= tf else "FP-Growth"
        sama = (len(res["Apriori"]["frequent"]) == len(res["FP-Growth"]["frequent"])
                and len(res["Apriori"]["rules"]) == len(res["FP-Growth"]["rules"]))
        db.save_comparison_result(
            ids[cepat], code, "Perbandingan", f"Perbandingan Apriori vs FP-Growth (min support {sup_pct:g}%)",
            f"Apriori: {len(res['Apriori']['frequent'])} frequent, {len(res['Apriori']['rules'])} aturan, {ta:.4f} detik. "
            f"FP-Growth: {len(res['FP-Growth']['frequent'])} frequent, {len(res['FP-Growth']['rules'])} aturan, {tf:.4f} detik. "
            f"Algoritma tercepat: {cepat}. " + ("Kedua algoritma menghasilkan frequent itemset dan aturan yang sama."
                                               if sama else "Jumlah hasil kedua algoritma berbeda."), cepat)
        for rec in mn.make_recommendations(res["FP-Growth"]["rules"]):
            db.save_comparison_result(ids["FP-Growth"], code, "Rekomendasi", rec["judul"], rec["deskripsi"])
    S["current_code"] = S["both_code"] = code
    S["aid_Apriori"], S["aid_FP-Growth"] = ids["Apriori"], ids["FP-Growth"]
    S["flash"] = f"Kedua algoritma selesai dijalankan (kode perbandingan {code})."
    st.rerun()


def render_perbedaan(ap, fg):
    """Tabel perbedaan Apriori vs FP-Growth + selisih waktu."""
    section("Perbedaan Apriori vs FP-Growth")
    if (ap["minimum_support"], ap["minimum_confidence"]) != (fg["minimum_support"], fg["minimum_confidence"]):
        st.warning("Parameter kedua hasil berbeda (min support / min confidence), "
                   "sehingga perbandingan ini kurang adil. Jalankan ulang dengan parameter yang sama.")
    st.dataframe(rp.comparison_df({"Apriori": ap, "FP-Growth": fg}), hide_index=True, width="stretch")
    ta, tf = ap["waktu_eksekusi"], fg["waktu_eksekusi"]
    cepat, lambat = ("Apriori", "FP-Growth") if ta <= tf else ("FP-Growth", "Apriori")
    rasio = max(ta, tf) / max(min(ta, tf), 1e-9)
    st.markdown(f'<div class="rec"><b>Selisih waktu</b><p>{cepat} lebih cepat {abs(ta - tf):.4f} detik '
                f'(±{rasio:.2f}× dibanding {lambat}).</p></div>', unsafe_allow_html=True)


OPSI_ANALISIS = ["Apriori", "FP-Growth", "Kedua Algoritma"]


def _kartu_jalankan(kunci, tombol):
    """Card kecil: parameter + tombol jalankan dalam satu baris."""
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1, 1, .7, 1.3])
        sup = c1.selectbox("Minimum Support", SUPPORT_OPT, index=SUPPORT_OPT.index(5),
                           format_func=lambda v: f"{v:g}%", key=f"sup_{kunci}")
        conf = c2.selectbox("Minimum Confidence", CONF_OPT, index=0,
                            format_func=lambda v: f"{v}%", key=f"conf_{kunci}")
        c3.text_input("Lift Ratio", value=">1", disabled=True, key=f"lift_{kunci}")
        c4.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        jalan = c4.button(tombol, key=f"run_{kunci}", type="primary", width="stretch")
    return jalan, sup, conf


def _ringkas(a):
    st.caption(f'Min support {a["minimum_support"]:g}% · min confidence {a["minimum_confidence"]:g}% · '
               f'lift > {a["minimum_lift"]:g} · {a["tanggal_analisis"].replace("T", " ")}')


def _tab_algoritma(name):
    """Satu algoritma: card kecil di atas, hasil lengkap (lebar penuh) di bawahnya."""
    jalan, sup, conf = _kartu_jalankan(name, f"Jalankan {name}")
    if jalan:
        jalankan_satu(name, sup, conf)
    aid = S.get(f"aid_{name}")
    if not aid:
        st.info("Belum ada hasil.")
        return
    a = db.get_analysis(aid)
    if not a:
        S.pop(f"aid_{name}", None)
        st.info("Hasil sudah dihapus dari riwayat.")
        return
    _ringkas(a)
    render_result(a, f"an_{aid}")


def _tab_kedua():
    """Kedua algoritma dijalankan sekaligus: tabel perbedaan, lalu hasil masing-masing."""
    jalan, sup, conf = _kartu_jalankan("kedua", "Jalankan Kedua Algoritma")
    if jalan:
        jalankan_analisis(sup, conf)
    code = S.get("both_code")
    if not code:
        st.info("Belum ada hasil.")
        return
    an = db.get_analysis_by_code(code)
    if not an:
        S.pop("both_code", None)
        st.info("Hasil sudah dihapus dari riwayat.")
        return
    ap, fg = an.get("Apriori"), an.get("FP-Growth")
    _ringkas(ap or fg)
    if ap and fg:
        render_perbedaan(ap, fg)
    if ap:
        section("Hasil Apriori")
        render_result(ap, f"both_{ap['analysis_id']}")
    if fg:
        section("Hasil FP-Growth")
        render_result(fg, f"both_{fg['analysis_id']}")


def page_analisis():
    topbar("Halaman Analisis Algoritma")
    if S.get("an_mode") not in OPSI_ANALISIS:
        S["an_mode"] = OPSI_ANALISIS[0]
    mode = tab_buttons(OPSI_ANALISIS, "an_mode")
    if mode == "Kedua Algoritma":
        _tab_kedua()
    else:
        _tab_algoritma(mode)


# =============================================================== perbandingan & rekomendasi
def page_perbandingan():
    topbar("Halaman Perbandingan Algoritma dan Rekomendasi Bisnis")
    codes = db.list_codes()
    if not codes:
        st.info("Belum ada hasil analisis. Jalankan analisis di halaman Analisis Algoritma.")
        return
    lab = {c["comparison_code"]: f'{c["comparison_code"]} · min support {c["sup"]:g}% · {c["tgl"].replace("T", " ")}'
           for c in codes}
    kode = st.selectbox("Pilih hasil analisis", list(lab), format_func=lab.get, key="cmp_code")
    an = db.get_analysis_by_code(kode)
    tab = tab_buttons(["Perbandingan", "Rekomendasi"], "cmp_tab")

    if tab == "Perbandingan":
        df = rp.comparison_df(an)
        st.dataframe(df, hide_index=True, width="stretch")
        res = db.get_results(kode, "Perbandingan")
        if res:
            st.markdown(f'<div class="rec"><b>Kesimpulan</b><p>{res[0]["deskripsi"]}</p></div>', unsafe_allow_html=True)
        section("Visualisasi Perbandingan")
        sm = db.all_analysis_summary()
        if sm.empty:
            st.info("No data to display")
            return
        # data terbaru untuk tiap kombinasi (algoritma, min support)
        sm = sm.sort_values("analysis_id").groupby(["nama_algoritma", "minimum_support"], as_index=False).last()
        sm = sm.sort_values("minimum_support")
        x = [f"{v:g}%" for v in sorted(sm["minimum_support"].unique())]

        def serie(col, name):
            d = sm[sm["nama_algoritma"] == name].set_index("minimum_support")[col]
            return [d.get(v) for v in sorted(sm["minimum_support"].unique())]

        clr = {"Apriori": sty.C["apriori"], "FP-Growth": sty.C["fpg"]}
        g1, g2, g3 = st.columns(3)
        for col, kolom, judul, jenis in [(g1, "jumlah_frequent_itemset", "Diagram Frequent Itemset", "bar"),
                                         (g2, "jumlah_aturan_asosiasi", "Diagram Aturan Asosiasi", "bar"),
                                         (g3, "waktu_eksekusi", "Grafik Waktu Eksekusi (detik)", "line")]:
            fig = go.Figure()
            for n in ("Apriori", "FP-Growth"):
                y = serie(kolom, n)
                if jenis == "bar":
                    fig.add_bar(x=x, y=y, name=n, marker_color=clr[n])
                else:
                    fig.add_scatter(x=x, y=y, name=n, mode="lines+markers", line=dict(color=clr[n], width=3))
            fig.update_xaxes(title="Minimum Support")
            col.plotly_chart(sty.plotly_layout(fig, judul), width="stretch")
    else:
        section("Rekomendasi Strategi Bisnis")
        rec = db.get_results(kode, "Rekomendasi")
        if not rec:
            st.info("Tidak ada aturan dengan confidence ≥ minimum dan lift > 1 pada analisis ini, "
                    "sehingga belum ada rekomendasi.")
        st.markdown('<div class="small-note">Disusun dari aturan asosiasi dengan lift tertinggi '
                    '(confidence ≥ minimum dan lift > 1).</div><br>', unsafe_allow_html=True)
        for r in rec:
            st.markdown(f'<div class="rec"><b>{r["judul"]}</b><p>{r["deskripsi"]}</p></div>', unsafe_allow_html=True)


# =============================================================== riwayat
PER_PAGE = 10


def page_riwayat():
    topbar("Halaman Riwayat Analisis")
    S.setdefault("hist_page", 1)
    c1, c2 = st.columns([3, 1])
    cari = c1.text_input("Cari riwayat", placeholder="Cari berdasarkan algoritma, tanggal, min support, atau kode…",
                         label_visibility="collapsed")
    if S.get("_cari") != cari:
        S["_cari"], S["hist_page"] = cari, 1
    rows, total = db.list_history(cari, PER_PAGE, (S["hist_page"] - 1) * PER_PAGE)
    if c2.button("Hapus Semua Riwayat", key="danger_hist", disabled=total == 0 and not cari, width="stretch"):
        confirm_delete("SELURUH riwayat analisis (beserta perbandingan & rekomendasi) akan dihapus permanen.",
                       lambda: f"{db.delete_all_analysis()} riwayat analisis berhasil dihapus.", ketik="HAPUS")

    st.markdown('<div class="subtle">Daftar Riwayat Analisis yang telah dilakukan</div>', unsafe_allow_html=True)
    w = [.5, 1.7, 1.3, 1.1, 1.2, .9, .9]
    with st.container(border=True):
        h = st.columns(w)
        for c, t in zip(h, ["No", "Tanggal Analisis", "Jenis Algoritma", "Min Support", "Min Confidence", "Detail", "Hapus"]):
            c.markdown(f'<div class="hdr-row">{t}</div>', unsafe_allow_html=True)
        if not rows:
            st.info("Belum ada riwayat analisis." if not cari else "Tidak ada riwayat yang cocok dengan pencarian.")
        for i, r in enumerate(rows, (S["hist_page"] - 1) * PER_PAGE + 1):
            c = st.columns(w)
            c[0].write(i)
            c[1].write(r["tanggal_analisis"].replace("T", " "))
            c[2].write(r["nama_algoritma"])
            c[3].write(f'{r["minimum_support"]:g}%')
            c[4].write(f'{r["minimum_confidence"]:g}%')
            if c[5].button("Detail", key=f"det_{r['analysis_id']}", width="stretch"):
                S["detail_id"] = r["analysis_id"]
                st.rerun()
            if c[6].button("Hapus", key=f"dangerrow_{r['analysis_id']}", width="stretch"):
                aid = r["analysis_id"]
                confirm_delete(f'Riwayat {r["nama_algoritma"]} tanggal {r["tanggal_analisis"].replace("T", " ")} akan dihapus.',
                               lambda aid=aid: "Riwayat analisis berhasil dihapus." if db.delete_analysis(aid) else "Data tidak ditemukan.")
    a_, b_ = (S["hist_page"] - 1) * PER_PAGE + 1, min(S["hist_page"] * PER_PAGE, total)
    p1, p2, p3 = st.columns([4, 1, 1])
    p1.markdown(f'Menampilkan {a_ if total else 0}-{b_} dari {total} data')
    pages = max(1, -(-total // PER_PAGE))
    if p2.button("◀ Sebelumnya", disabled=S["hist_page"] <= 1, width="stretch"):
        S["hist_page"] -= 1
        st.rerun()
    if p3.button("Berikutnya ▶", disabled=S["hist_page"] >= pages, width="stretch"):
        S["hist_page"] += 1
        st.rerun()

    if S.get("detail_id"):
        a = db.get_analysis(S["detail_id"])
        if a:
            section(f'Detail Analisis — {a["nama_algoritma"]} (kode {a["comparison_code"]})')
            st.caption(f'min support {a["minimum_support"]:g}% · min confidence {a["minimum_confidence"]:g}% · '
                       f'lift > {a["minimum_lift"]:g} · {a["tanggal_analisis"].replace("T", " ")}')
            render_result(a, f"hist_{a['analysis_id']}")
            if st.button("Tutup Detail"):
                S.pop("detail_id")
                st.rerun()
        else:
            S.pop("detail_id")


# =============================================================== ekspor laporan
def page_ekspor():
    topbar("Halaman Ekspor Laporan")
    st.markdown('<div class="subtle">Pilih Jenis Laporan yang ingin di Ekspor</div>', unsafe_allow_html=True)
    s = db.transaction_stats()
    periode = ["Semua Periode"]
    if s["b"]:
        bulan = pd.period_range(s["a"], s["z"], freq="M").astype(str).tolist()
        periode += bulan
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        jenis = c1.selectbox("Jenis Laporan", rp.JENIS_LAPORAN)
        c2.text_input("Format", value="excel", disabled=True)
        per = c3.selectbox("Periode Data", periode)
        start = end = None
        if per != "Semua Periode":
            p = pd.Period(per)
            start, end = p.start_time.date().isoformat(), p.end_time.date().isoformat()
        df = db.get_transactions(start, end)
        c4, c5, c6 = st.columns(3)
        c4.text_input("Tanggal Ekspor", value=date.today().strftime("%d-%m-%Y"), disabled=True)
        c5.text_input("Jumlah Transaksi", value=str(df["no_transaksi"].nunique()), disabled=True)
        c6.text_input("Jumlah Produk", value=str(df["nama_produk"].nunique()), disabled=True)
        st.caption("Periode data berlaku untuk data transaksi; hasil analisis yang diekspor adalah analisis terakhir.")
    if jenis in ("Laporan Lengkap", "Data Transaksi") and df.empty:
        st.warning("Tidak ada data transaksi pada periode ini.")
    data, warn = rp.build_excel(jenis, start, end)
    for w_ in warn:
        st.warning(w_)
    if data:
        st.download_button("Ekspor ke Excel", data, file_name=rp.file_name(jenis), type="primary",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# =============================================================== router
if not S["user"]:
    page_login()
else:
    sidebar()
    {"Dashboard": page_dashboard, "Kelola Data": page_kelola, "Analisis Algoritma": page_analisis,
     "Perbandingan dan Rekomendasi": page_perbandingan, "Riwayat Analisis": page_riwayat,
     "Ekspor Laporan": page_ekspor}[S["page"]]()