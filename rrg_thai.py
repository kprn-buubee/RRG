#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RRG — Relative Rotation Graph สำหรับหุ้นไทย  (แผนที่ตามรอยเงิน)
================================================================
คำนวณ RS-Ratio (แกนนอน) และ RS-Momentum (แกนตั้ง) ของกลุ่มธีม/หุ้นรายตัว
เทียบกับ benchmark (ค่าเริ่มต้น = ^SET.BK) แล้ว export เป็น HTML dashboard
แบบ interactive ไฟล์เดียว เปิดด้วยเบราว์เซอร์ได้เลย

วิธีใช้
-------
  pip install yfinance pandas numpy

  # โหมดธีม (ค่าเริ่มต้น) — รวมหุ้นเป็นดัชนีกลุ่มแบบถ่วงน้ำหนักเท่ากัน
  python rrg_thai.py

  # เทียบกับ SET50 แทน + หางย้อนหลัง 15 วัน
  python rrg_thai.py --benchmark ^SET50.BK --tail 15

  # โหมดหุ้นรายตัว (watchlist ของเราเอง)
  python rrg_thai.py --tickers PTT,KBANK,DELTA,CPALL,GULF,BDMS

  # รายสัปดาห์ (สัญญาณช้าลง เหมาะกับ position)
  python rrg_thai.py --interval 1wk --window 30 --mom 4 --tail 8

  # ลองรันโดยไม่ต่อเน็ต (ข้อมูลจำลอง) เพื่อดูหน้าตา dashboard
  python rrg_thai.py --demo

  # แท็บ Breadth (% หุ้นเหนือ SMA200/50) สร้างให้อัตโนมัติจากหุ้นทุกตัวในแดชบอร์ด
  # ถ้าอยากใช้ฐานอื่น เช่น SET100 ให้ทำไฟล์รายชื่อ (คั่นด้วย comma หรือขึ้นบรรทัดใหม่)
  python rrg_thai.py --breadth-list set100.txt

สูตรที่ใช้
----------
  RS            = 100 × ราคากลุ่ม / ราคาดัชนี
  RS-Ratio      = 100 + z-score(RS, window)          → "ตำแหน่ง" ความแข็งแกร่ง
  RS-Momentum   = 100 + z-score(ΔRS-Ratio(mom), window) → "ทิศทาง" ของความแข็งแกร่ง

  z-score ทำให้ทุกกลุ่มอยู่บนสเกลเดียวกัน (ปกติ 98–102) เทียบกันได้ตรง ๆ
  เป็นวิธี normalize แบบเปิด ที่ให้ผลใกล้เคียง JdK RS-Ratio/RS-Momentum ต้นฉบับ
  (ของ Julius de Kempenaer ซึ่งเป็นสูตรลิขสิทธิ์ ไม่ได้เปิดเผยรายละเอียด)

  จตุภาค:  ขวาบน = นำตลาด | ขวาล่าง = อ่อนแรง
           ซ้ายล่าง = ตามหลัง | ซ้ายบน = กำลังฟื้น
  วงจรเงินไหลปกติ: ฟื้น → นำ → อ่อนแรง → ตามหลัง → ฟื้น (หมุนทวนเข็มนาฬิกา)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════════
#  ชุดที่ 1: "สายสัมพันธ์" — หุ้นที่เคลื่อนไหวไปด้วยกัน (ค่าเริ่มต้น)
#  กลุ่มบริษัทเดียวกัน / ซัพพลายเชนเดียวกัน / ธีมข่าวเดียวกัน
#  โครงสร้าง: "ชื่อธีม": ("หมวดใหญ่", [รายชื่อหุ้น])
#  หุ้นตัวเดียวอยู่ได้หลายธีม (เช่น GULF, KBANK, SCC) ไม่ต้องกังวล
# ══════════════════════════════════════════════════════════════════════
PAIR_THEMES: dict[str, tuple[str, list[str]]] = {
    # ── การเงิน ──────────────────────────────────────────────────
    "สาย JMART":            ("การเงิน", ["JMART", "JMT", "SINGER", "SGC", "SCAP"]),
    "จำนำทะเบียน":          ("การเงิน", ["MTC", "SAWAD", "TIDLOR"]),
    "แบงก์ใหญ่":            ("การเงิน", ["KBANK", "BBL", "SCB"]),
    "แบงก์เล็ก & ปันผล":    ("การเงิน", ["TISCO", "KKP", "TCAP", "CREDIT"]),
    "ประกัน":               ("การเงิน", ["BLA", "TLI", "TQM"]),

    # ── พลังงาน & สาธารณูปโภค ────────────────────────────────────
    # "เครือกัลฟ์" = เครือข่ายการลงทุนของ สารัชถ์ รัตนาวะดี / กลุ่มกัลฟ์
    # เชื่อมกันด้วยโครงสร้างผู้ถือหุ้น ไม่ใช่วัฏจักรธุรกิจร่วมกัน
    "โรงไฟฟ้า":             ("พลังงาน & สาธารณูปโภค", ["GULF", "GUNKUL", "BGRIM", "GPSC"]),
    "น้ำ & สาธารณูปโภค":     ("พลังงาน & สาธารณูปโภค", ["WHAUP", "EASTW", "TTW"]),
    "กลุ่ม ปตท.":            ("พลังงาน & สาธารณูปโภค", ["PTT", "PTTEP", "PTTGC"]),
    "เคมีภัณฑ์":            ("พลังงาน & สาธารณูปโภค", ["PTTGC", "IVL", "UKEM"]),
    "เครือกัลฟ์ · แกนหลัก":   ("พลังงาน & สาธารณูปโภค", ["GULF", "ADVANC", "KBANK"]),

    # ── อสังหา & ก่อสร้าง ────────────────────────────────────────
    "กลุ่ม SCG":            ("อสังหา & ก่อสร้าง", ["SCC", "SCGP", "SJWD", "SCGD"]),
    "กลุ่ม ช.การช่าง":       ("อสังหา & ก่อสร้าง", ["CK", "CKP", "BEM", "TTW"]),
    "อสังหาฯ ที่อยู่อาศัย":   ("อสังหา & ก่อสร้าง", ["AP", "SPALI", "QH", "LH"]),
    "นิคมอุตสาหกรรม":       ("อสังหา & ก่อสร้าง", ["WHA", "AMATA", "ROJNA", "PIN"]),
    "วางระบบ & โครงข่าย":    ("อสังหา & ก่อสร้าง", ["INSET", "ASEFA", "ALT", "ITEL", "CNT"]),
    "เครือกัลฟ์ · ลงทุนนอกกลุ่ม": ("อสังหา & ก่อสร้าง", ["GULF", "MINT", "KBANK", "CIVIL"]),

    # ── บริโภค ───────────────────────────────────────────────────
    "ค้าปลีกวัสดุ-ตกแต่งบ้าน": ("บริโภค", ["HMPRO", "GLOBAL", "DOHOME"]),
    "เครื่องดื่ม":           ("บริโภค", ["OSP", "CBG", "ICHI"]),
    "สุขภาพ & เครื่องดื่ม":   ("บริโภค", ["MEGA", "CBG", "OSP"]),
    "ไก่ & ปศุสัตว์":        ("บริโภค", ["TFG", "GFPT", "BTG", "CPF"]),
    "อาหารทะเล & อาหารสัตว์": ("บริโภค", ["TU", "ITC", "ASIAN", "AAI"]),

    # ── บริการ & ท่องเที่ยว ──────────────────────────────────────
    "ท่องเที่ยว":            ("บริการ & ท่องเที่ยว", ["AOT", "MINT", "AWC", "ERW", "SHR"]),
    "สื่อนอกบ้าน":           ("บริการ & ท่องเที่ยว", ["VGI", "PLANB"]),
    "เรือเทกอง":            ("บริการ & ท่องเที่ยว", ["PSL", "TTA"]),
    "คาร์โกทางอากาศ":        ("บริการ & ท่องเที่ยว", ["III", "ANI", "SKY"]),
    "มือถือ":               ("บริการ & ท่องเที่ยว", ["ADVANC", "TRUE"]),

    # ── สุขภาพ ───────────────────────────────────────────────────
    "รพ. พรีเมียม":         ("สุขภาพ", ["BDMS", "BH"]),
    "รพ. ประกันสังคม":       ("สุขภาพ", ["BCH", "RJH", "VIH", "CHG"]),

    # ── เทคโนโลยี ────────────────────────────────────────────────
    "ไอทีโครงการรัฐ":        ("เทคโนโลยี", ["SKY", "AIT", "TKC", "PLANB"]),
    "ดิจิทัลภาครัฐ":         ("เทคโนโลยี", ["DITTO", "TEAMG"]),
    "อิเล็กทรอนิกส์ AI":      ("เทคโนโลยี", ["DELTA", "CCET"]),
    "PCB & ชิ้นส่วน":        ("เทคโนโลยี", ["KCE", "HANA", "FORTH"]),
    "สาย FORTH":            ("เทคโนโลยี", ["FORTH", "FSMART"]),

    # ── อุตสาหกรรม ───────────────────────────────────────────────
    "ชิ้นส่วนยานยนต์":       ("อุตสาหกรรม", ["AH", "SAT"]),
}

# ══════════════════════════════════════════════════════════════════════
#  ชุดที่ 2: sector มาตรฐาน  (เรียกด้วย --universe sector)
# ══════════════════════════════════════════════════════════════════════
SECTOR_THEMES: dict[str, tuple[str, list[str]]] = {
    "ธนาคาร":             ("การเงิน", ["KBANK", "SCB", "BBL", "KTB", "TTB", "TISCO", "KKP"]),
    "เงินทุน & สินเชื่อ":    ("การเงิน", ["MTC", "SAWAD", "TIDLOR", "KTC", "AEONTS", "SAK"]),
    "ประกัน":              ("การเงิน", ["TLI", "BLA", "TQM", "TIPH"]),
    "พลังงาน & โรงกลั่น":   ("พลังงาน & สาธารณูปโภค", ["PTT", "PTTEP", "TOP", "BCP", "SPRC", "IRPC", "BANPU"]),
    "โรงไฟฟ้า":            ("พลังงาน & สาธารณูปโภค", ["GULF", "GPSC", "EGCO", "RATCH", "BGRIM", "CKP"]),
    "ปิโตรเคมี":           ("พลังงาน & สาธารณูปโภค", ["PTTGC", "IVL", "TPIPL"]),
    "วัสดุก่อสร้าง":        ("อสังหา & ก่อสร้าง", ["SCC", "SCCC", "TASCO", "DCC", "SCGP"]),
    "รับเหมาก่อสร้าง":      ("อสังหา & ก่อสร้าง", ["STECON", "CK", "PYLON", "SEAFCO"]),
    "อสังหาฯ ที่อยู่อาศัย":   ("อสังหา & ก่อสร้าง", ["LH", "AP", "SPALI", "SIRI", "ORI", "QH", "PSH"]),
    "นิคมอุตสาหกรรม":      ("อสังหา & ก่อสร้าง", ["AMATA", "WHA", "ROJNA", "PIN"]),
    "ค้าปลีก":             ("บริโภค", ["CPALL", "CRC", "HMPRO", "GLOBAL", "BJC", "COM7", "DOHOME"]),
    "อาหาร & เครื่องดื่ม":   ("บริโภค", ["CPF", "TU", "GFPT", "OSP", "CBG", "TFG"]),
    "ท่องเที่ยว & โรงแรม":   ("บริการ & ท่องเที่ยว", ["MINT", "CENTEL", "ERW", "AWC", "SHR"]),
    "การบิน & ขนส่ง":      ("บริการ & ท่องเที่ยว", ["AOT", "AAV", "BA", "BEM", "BTS", "PSL", "RCL"]),
    "โรงพยาบาล":          ("สุขภาพ", ["BDMS", "BH", "BCH", "CHG", "PR9", "RAM"]),
    "ICT & สื่อสาร":       ("เทคโนโลยี", ["ADVANC", "TRUE", "JAS", "SAMART"]),
    "อิเล็กทรอนิกส์":       ("เทคโนโลยี", ["DELTA", "KCE", "HANA", "CCET"]),
    "สื่อ & บันเทิง":       ("บริการ & ท่องเที่ยว", ["VGI", "PLANB", "MAJOR", "WORK", "ONEE"]),
    "ชิ้นส่วนยานยนต์":      ("อุตสาหกรรม", ["STANLY", "SAT", "AH", "PCSGH"]),
    "บรรจุภัณฑ์ & กระดาษ":  ("อุตสาหกรรม", ["TPBI", "SFLEX", "UTP"]),
}

# ══════════════════════════════════════════════════════════════════════
#  ชุดที่ 3: "ห่วงโซ่มูลค่า"  (เรียกด้วย --universe chain)
#  ซอยตามชั้นของห่วงโซ่ แบบเดียวกับ dashboard ฝั่ง US ที่แตกเรื่องชิป
#  เป็น GPU / HBM / เครื่องจักร แยกกัน เพราะแต่ละชั้นได้เงินไม่พร้อมกัน
#  เลข ①②③ คือลำดับที่เงินควรไหลตามทฤษฎี — ใช้เทียบว่าชั้นไหนนำ ชั้นไหนตกค้าง
# ══════════════════════════════════════════════════════════════════════
CHAIN_THEMES: dict[str, tuple[str, list[str]]] = {
    # ── ไฟฟ้า & ดาต้าเซ็นเตอร์ ────────────────────────────────────
    "ไฟฟ้า ① ผลิตไฟ":        ("ไฟฟ้า & ดาต้าเซ็นเตอร์", ["GULF", "GPSC", "BGRIM", "EGCO", "RATCH", "GUNKUL"]),
    "ไฟฟ้า ② วางระบบ-สายส่ง": ("ไฟฟ้า & ดาต้าเซ็นเตอร์", ["INSET", "ITEL", "ALT", "ASEFA", "CNT"]),
    "ไฟฟ้า ③ อุปกรณ์-ชิ้นส่วน": ("ไฟฟ้า & ดาต้าเซ็นเตอร์", ["DELTA", "KCE", "HANA", "CCET"]),
    "ไฟฟ้า ④ โครงข่าย-ผู้ใช้":  ("ไฟฟ้า & ดาต้าเซ็นเตอร์", ["ADVANC", "TRUE"]),
    "ไฟฟ้า ⑤ ไอทีโซลูชัน":    ("ไฟฟ้า & ดาต้าเซ็นเตอร์", ["AIT", "TKC", "SKY", "DITTO"]),

    # ── ท่องเที่ยว ────────────────────────────────────────────────
    "ท่องเที่ยว ① ประตูเข้า":   ("ห่วงโซ่ท่องเที่ยว", ["AOT", "AAV", "BA"]),
    "ท่องเที่ยว ② ที่พัก":      ("ห่วงโซ่ท่องเที่ยว", ["MINT", "ERW", "CENTEL", "SHR", "AWC"]),
    "ท่องเที่ยว ③ ใช้จ่ายปลายทาง": ("ห่วงโซ่ท่องเที่ยว", ["CPALL", "CRC", "BJC", "COM7"]),
    "ท่องเที่ยว ④ ขนส่งในเมือง":  ("ห่วงโซ่ท่องเที่ยว", ["BTS", "BEM"]),
    "ท่องเที่ยว ⑤ การแพทย์":    ("ห่วงโซ่ท่องเที่ยว", ["BDMS", "BH"]),

    # ── อสังหา & ก่อสร้าง ────────────────────────────────────────
    "อสังหา ① ที่ดิน-นิคม":     ("ห่วงโซ่อสังหา", ["WHA", "AMATA", "ROJNA", "PIN"]),
    "อสังหา ② รับเหมา":        ("ห่วงโซ่อสังหา", ["STECON", "CK", "CIVIL", "SEAFCO", "PYLON"]),
    "อสังหา ③ วัสดุก่อสร้าง":    ("ห่วงโซ่อสังหา", ["SCC", "SCGD", "TASCO", "DCC", "TPIPL"]),
    "อสังหา ④ ผู้พัฒนา":       ("ห่วงโซ่อสังหา", ["AP", "SPALI", "LH", "QH", "SIRI", "ORI"]),
    "อสังหา ⑤ สินเชื่อบ้าน":    ("ห่วงโซ่อสังหา", ["KBANK", "SCB", "BBL", "KTB", "TTB"]),
    "อสังหา ⑥ ตกแต่ง-ค้าปลีกวัสดุ": ("ห่วงโซ่อสังหา", ["HMPRO", "GLOBAL", "DOHOME"]),

    # ── อาหาร ────────────────────────────────────────────────────
    "อาหาร ① วัตถุดิบ-อาหารสัตว์": ("ห่วงโซ่อาหาร", ["TVO", "ASIAN", "AAI"]),
    "อาหาร ② ฟาร์ม-ปศุสัตว์":   ("ห่วงโซ่อาหาร", ["CPF", "GFPT", "TFG", "BTG"]),
    "อาหาร ③ แปรรูปส่งออก":    ("ห่วงโซ่อาหาร", ["TU", "ITC", "ASIAN"]),
    "อาหาร ④ ค้าปลีก-ร้านอาหาร": ("ห่วงโซ่อาหาร", ["CPALL", "M", "OSP", "CBG"]),
}

UNIVERSES = {"pairs": PAIR_THEMES, "sector": SECTOR_THEMES, "chain": CHAIN_THEMES}

PALETTE = [
    "#22d3ee", "#a78bfa", "#f472b6", "#fbbf24", "#34d399", "#60a5fa",
    "#fb923c", "#e879f9", "#4ade80", "#f87171", "#38bdf8", "#c084fc",
    "#facc15", "#2dd4bf", "#fb7185", "#818cf8", "#a3e635", "#f9a8d4",
    "#5eead4", "#fdba74",
]

GROUP_ORDER = ["การเงิน", "พลังงาน & สาธารณูปโภค", "อสังหา & ก่อสร้าง", "บริโภค",
               "บริการ & ท่องเที่ยว", "สุขภาพ", "เทคโนโลยี", "อุตสาหกรรม",
               "ไฟฟ้า & ดาต้าเซ็นเตอร์", "ห่วงโซ่ท่องเที่ยว", "ห่วงโซ่อสังหา", "ห่วงโซ่อาหาร",
               "หุ้นรายตัว"]


# ══════════════════════════════════════════════════════════════════════
#  ดึงข้อมูลราคา
# ══════════════════════════════════════════════════════════════════════
def yahoo_symbol(t: str) -> str:
    """PTT → PTT.BK ; ^SET.BK คงเดิม ; ระบุ .BK มาแล้วก็คงเดิม"""
    t = t.strip().upper()
    if t.startswith("^") or "." in t:
        return t
    return f"{t}.BK"


def fetch_prices(symbols: list[str], period: str, interval: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("ไม่พบ yfinance — ติดตั้งด้วย:  pip install yfinance")

    print(f"กำลังดึงข้อมูล {len(symbols)} สัญลักษณ์ ...", file=sys.stderr)
    raw = yf.download(symbols, period=period, interval=interval,
                      auto_adjust=True, progress=False, threads=True)
    if raw is None or len(raw) == 0:
        sys.exit("ดึงข้อมูลไม่สำเร็จ — ตรวจการเชื่อมต่อหรือชื่อสัญลักษณ์")

    def pick(field: str) -> pd.DataFrame:
        if isinstance(raw.columns, pd.MultiIndex):
            out = raw[field]
        else:
            out = raw[[field]]
        if isinstance(out, pd.Series):
            out = out.to_frame(symbols[0])
        if len(symbols) == 1 and list(out.columns) == [field]:
            out.columns = symbols
        return out.sort_index()

    return pick("Close"), pick("Volume")


def demo_prices(symbols: list[str], n: int = 320, seed: int = 7) -> pd.DataFrame:
    """ข้อมูลจำลองสำหรับดูหน้าตา dashboard โดยไม่ต้องต่อเน็ต"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    n = len(idx)  # pandas บางเวอร์ชันคืนจำนวนไม่ตรง periods พอดี
    market = np.cumsum(rng.normal(0.0004, 0.009, n))
    out = {}
    for i, s in enumerate(symbols):
        beta = rng.uniform(0.6, 1.4)
        drift = rng.normal(0.0002, 0.0009)
        cyc = 0.10 * np.sin(np.linspace(rng.uniform(0, 6), rng.uniform(6, 14), n))
        idio = np.cumsum(rng.normal(drift, 0.013, n))
        out[s] = 100 * np.exp(beta * market + idio + cyc)
    return pd.DataFrame(out, index=idx)


# ══════════════════════════════════════════════════════════════════════
#  คำนวณ RRG
# ══════════════════════════════════════════════════════════════════════
def composite_index(px: pd.DataFrame) -> pd.Series:
    """รวมหุ้นหลายตัวเป็นดัชนีกลุ่มแบบถ่วงน้ำหนักเท่ากัน (rebase 100)"""
    px = px.dropna(axis=1, how="all").ffill()
    if px.empty:
        return pd.Series(dtype=float)
    base = px.apply(lambda c: c.loc[c.first_valid_index()])
    return (px / base).mean(axis=1, skipna=True) * 100


def zscore(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(win, min_periods=max(5, win // 3)).mean()
    sd = s.rolling(win, min_periods=max(5, win // 3)).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def rrg_coords(series: pd.Series, bench: pd.Series,
               window: int, mom: int, smooth: int) -> pd.DataFrame:
    df = pd.concat([series.rename("s"), bench.rename("b")], axis=1).dropna()
    if len(df) < window + mom + 5:
        return pd.DataFrame(columns=["ratio", "momentum"])

    rs = 100.0 * df["s"] / df["b"]
    ratio = 100.0 + zscore(rs, window).ewm(span=smooth, adjust=False).mean()
    roc = ratio.diff(mom)
    momentum = 100.0 + zscore(roc, window).ewm(span=smooth, adjust=False).mean()
    return pd.concat([ratio.rename("ratio"), momentum.rename("momentum")], axis=1).dropna()


def quadrant(ratio: float, momentum: float) -> str:
    if ratio >= 100 and momentum >= 100:
        return "leading"
    if ratio >= 100:
        return "weakening"
    if momentum >= 100:
        return "improving"
    return "lagging"


# ══════════════════════════════════════════════════════════════════════
#  Breadth — % หุ้นที่ปิดเหนือเส้นค่าเฉลี่ย
# ══════════════════════════════════════════════════════════════════════
def build_breadth(px: pd.DataFrame, syms: list[str], bench: pd.Series,
                  n_long: int, n_short: int, label: str, keep: int = 400) -> dict | None:
    cols = [c for c in dict.fromkeys(syms) if c in px.columns and px[c].notna().sum() > n_long]
    if len(cols) < 5:
        return None
    p = px[cols].ffill(limit=5)

    def above_sma(n: int):
        sma = p.rolling(n, min_periods=n).mean()
        valid = sma.notna() & p.notna()
        above = (p > sma) & valid
        cnt = valid.sum(axis=1)
        return above.sum(axis=1) / cnt.replace(0, np.nan) * 100, above, valid, cnt

    pa, above_a, valid_a, cnt_a = above_sma(n_long)
    pb, *_ = above_sma(n_short)
    ok = cnt_a >= max(5, len(cols) // 2)          # ตัดช่วงต้นที่ SMA ยังคำนวณได้ไม่ครบ
    pa = pa[ok].tail(keep)
    if len(pa) < 2:
        return None
    idx = pa.index
    pb = pb.reindex(idx)
    bx = bench.reindex(idx).ffill()

    now, valid_now = above_a.loc[idx[-1]], valid_a.loc[idx[-1]]
    look = idx[-6] if len(idx) > 6 else idx[0]
    prev, valid_prev = above_a.loc[look], valid_a.loc[look]
    short = lambda c: c.replace(".BK", "")
    up = [short(c) for c in cols if valid_now[c] and valid_prev[c] and now[c] and not prev[c]]
    down = [short(c) for c in cols if valid_now[c] and valid_prev[c] and prev[c] and not now[c]]

    themes = []
    for name, (_, members) in SECTOR_THEMES.items():
        ms = [yahoo_symbol(m) for m in members if yahoo_symbol(m) in cols and valid_now[yahoo_symbol(m)]]
        if len(ms) < 2:
            continue
        ab = [short(m) for m in ms if now[m]]
        themes.append({"name": name, "pct": round(100 * len(ab) / len(ms), 1),
                       "n": len(ab), "of": len(ms), "above": ab,
                       "below": [short(m) for m in ms if not now[m]]})
    themes.sort(key=lambda t: (-t["pct"], t["name"]))

    r = lambda v: None if pd.isna(v) else round(float(v), 1)
    return {
        "label": label, "lenA": n_long, "lenB": n_short,
        "total": int(valid_now.sum()), "above": int(now.sum()),
        "dates": [d.strftime("%Y-%m-%d") for d in idx],
        "pA": [r(v) for v in pa], "pB": [r(v) for v in pb],
        "index": [None if pd.isna(v) else round(float(v), 2) for v in bx],
        "crossUp": up, "crossDown": down, "themes": themes,
    }


# ══════════════════════════════════════════════════════════════════════
#  HTML
# ══════════════════════════════════════════════════════════════════════
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root{
    --bg:#07060f; --bg2:#0d0b1c; --panel:rgba(255,255,255,.035);
    --stroke:rgba(255,255,255,.10); --ink:#eef1ff; --muted:#8b8fb4;
    --lead:#34d399; --weak:#fbbf24; --lag:#fb7185; --imp:#60a5fa;
    --font:'Sarabun','IBM Plex Sans Thai','Noto Sans Thai',-apple-system,'Segoe UI',sans-serif;
    --mono:'JetBrains Mono','SF Mono',ui-monospace,'Cascadia Code',monospace;
  }
  *{box-sizing:border-box}
  body{margin:0;padding:22px;background:
        radial-gradient(1100px 620px at 12% -12%,#241a4d 0%,transparent 62%),
        radial-gradient(900px 520px at 96% 8%,#10284a 0%,transparent 58%),var(--bg);
       color:var(--ink);font-family:var(--font);-webkit-font-smoothing:antialiased}
  .eyebrow{font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:#7dd3fc;font-weight:700}
  h1{margin:6px 0 2px;font-size:clamp(24px,3.4vw,40px);line-height:1.1;letter-spacing:-.02em;
     background:linear-gradient(92deg,#67e8f9,#a78bfa 55%,#f0abfc);-webkit-background-clip:text;
     background-clip:text;color:transparent}
  .sub{color:var(--muted);font-size:13px;margin-top:6px}
  .bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:18px 0 12px}
  .bar .lbl{font-size:11px;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin-right:2px}
  .chip{border:1px solid var(--stroke);background:var(--panel);color:var(--ink);border-radius:999px;
        padding:7px 15px;font:inherit;font-size:13px;cursor:pointer;transition:.16s;backdrop-filter:blur(8px)}
  .chip:hover{border-color:rgba(255,255,255,.3)}
  .chip[aria-pressed="true"]{color:#07060f;font-weight:700}
  .chip:focus-visible{outline:2px solid #67e8f9;outline-offset:2px}
  .layout{display:grid;grid-template-columns:minmax(0,1fr) 330px;gap:16px;align-items:start}
  .card{border:1px solid var(--stroke);background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.015));
        border-radius:18px;backdrop-filter:blur(10px)}
  #plot{padding:8px}
  svg{display:block;width:100%;height:auto;overflow:visible}
  .counts{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px}
  .count{padding:12px 8px;text-align:center;border-radius:14px}
  .count b{display:block;font-family:var(--mono);font-size:26px;line-height:1.1}
  .count span{font-size:10px;letter-spacing:.14em;color:var(--muted);text-transform:uppercase}
  .sec{padding:14px 16px;margin-bottom:12px}
  .sec h3{margin:0 0 10px;font-size:12px;letter-spacing:.16em;text-transform:uppercase}
  .row{display:flex;align-items:center;gap:9px;padding:5px 0;font-size:13.5px}
  .row .dot{width:9px;height:9px;border-radius:50%;flex:none}
  .row .nm{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .row .val{font-family:var(--mono);font-size:11.5px;color:var(--muted);flex:none}
  .empty{color:var(--muted);font-size:12.5px;padding:2px 0}
  footer{color:var(--muted);font-size:11.5px;margin-top:16px;line-height:1.7}
  [hidden]{display:none!important}
  #br-plot{padding:8px;position:relative}
  .legend{display:flex;flex-wrap:wrap;gap:14px;padding:8px 12px 0;font-size:12px;color:var(--muted)}
  .legend i{display:inline-block;width:18px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
  .tip{position:absolute;pointer-events:none;background:rgba(13,11,28,.92);border:1px solid var(--stroke);
       border-radius:10px;padding:8px 10px;font-size:12px;line-height:1.6;white-space:nowrap;display:none}
  .tip b{font-family:var(--mono)}
  .tiles{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px}
  .tile{padding:12px;border-radius:14px;border:1px solid var(--stroke);background:var(--panel)}
  .tile b{display:block;font-family:var(--mono);font-size:26px;line-height:1.15}
  .tile span{font-size:11px;color:var(--muted)}
  .meter{flex:none;width:90px;height:6px;border-radius:3px;background:rgba(255,255,255,.08);overflow:hidden}
  .meter i{display:block;height:100%;border-radius:3px}
  .tick{font-family:var(--mono);font-size:12px;color:var(--ink);line-height:1.9}
  @media (max-width:960px){.layout{grid-template-columns:1fr}}
  @media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
</head>
<body>
<header>
  <div class="eyebrow">__EYEBROW__</div>
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>
  <div class="sub" id="meta"></div>
</header>

<div class="bar" id="bar-tabs"></div>

<section id="tab-rrg">
<div class="bar" id="bar-views"></div>
<div class="bar" id="bar-view"></div>
<div class="bar" id="bar-groups"></div>

<div class="layout">
  <div class="card" id="plot"></div>
  <aside>
    <div class="counts" id="counts"></div>
    <div id="lists"></div>
  </aside>
</div>

<footer>
  แกนนอน RS-Ratio = ตำแหน่งความแข็งแกร่งเทียบดัชนี · แกนตั้ง RS-Momentum = ทิศทางของความแข็งแกร่ง ·
  ทั้งสองแกน normalize เป็น z-score รอบค่า 100<br>
  วงจรปกติหมุนทวนเข็มนาฬิกา: กำลังฟื้น → นำตลาด → อ่อนแรง → ตามหลัง ·
  เส้นหางยาว = หมุนเร็ว, หางสั้น/ขด = ยังไม่มีทิศทาง<br>
  ข้อมูลเพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน
</footer>
</section>

<section id="tab-breadth" hidden>
<div class="bar" id="bar-br"></div>
<div class="layout">
  <div class="card" id="br-plot"></div>
  <aside id="br-side"></aside>
</div>
<footer id="br-foot"></footer>
</section>

<script>
const DATA = __DATA__;

const QUAD = {
  leading:   {th:'นำตลาด',   en:'LEADING',   raw:'#34d399'},
  weakening: {th:'อ่อนแรง',  en:'WEAKENING', raw:'#fbbf24'},
  lagging:   {th:'ตามหลัง',  en:'LAGGING',   raw:'#fb7185'},
  improving: {th:'กำลังฟื้น', en:'IMPROVING', raw:'#60a5fa'}
};

const state = {
  u: DATA.defaultUniverse,
  b: DATA.defaultBenchmark,
  tail: DATA.defaultTail,
  labels: 'leaders',
  groups: new Set()
};

const V = () => DATA.views[state.u + '|' + state.b] || {groups: [], groupColors: {}, items: []};
const benchLabel = () => (DATA.benchmarks.find(x => x.key === state.b) || {}).label || '';
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const visible = () => V().items.filter(it => state.groups.has(it.group));
const resetGroups = () => { state.groups = new Set(V().groups); };
resetGroups();

/* ── แถบควบคุม ─────────────────────────────────────── */
function chip(label, active, onClick, color){
  const b = document.createElement('button');
  b.className = 'chip'; b.textContent = label;
  b.setAttribute('aria-pressed', active ? 'true' : 'false');
  if (color){
    b.style.borderColor = active ? color : color + '66';
    b.style.color = active ? '#07060f' : color;
    b.style.background = active ? color : 'transparent';
    if (active) b.style.boxShadow = '0 0 18px ' + color + '55';
  } else if (active){
    b.style.background = '#67e8f9'; b.style.borderColor = '#67e8f9';
    b.style.color = '#07060f'; b.style.boxShadow = '0 0 18px #67e8f955';
  }
  b.onclick = onClick;
  return b;
}
function lbl(text, host, gap){
  const e = document.createElement('span');
  e.className = 'lbl'; e.textContent = text;
  if (gap) e.style.marginLeft = '12px';
  host.appendChild(e);
}

function renderBars(){
  const v0 = document.getElementById('bar-views');
  v0.innerHTML = '';
  lbl('ชุดธีม', v0);
  DATA.universes.forEach(u => v0.appendChild(
    chip(u.label, state.u === u.key, () => { state.u = u.key; resetGroups(); render(); })));
  lbl('เทียบกับ', v0, true);
  DATA.benchmarks.forEach(b => v0.appendChild(
    chip(b.label, state.b === b.key, () => { state.b = b.key; resetGroups(); render(); })));

  const v = document.getElementById('bar-view');
  v.innerHTML = '';
  lbl('หาง', v);
  DATA.tailOptions.forEach(t => v.appendChild(
    chip(t + ' แท่ง', state.tail === t, () => { state.tail = t; render(); })));
  lbl('ป้ายชื่อ', v, true);
  [['all','ทั้งหมด'],['leaders','เฉพาะผู้นำ'],['none','ไม่แสดง']].forEach(([k, t]) =>
    v.appendChild(chip(t, state.labels === k, () => { state.labels = k; render(); })));

  const g = document.getElementById('bar-groups');
  g.innerHTML = '';
  V().groups.forEach(name => g.appendChild(
    chip(name, state.groups.has(name), () => {
      state.groups.has(name) ? state.groups.delete(name) : state.groups.add(name);
      render();
    }, V().groupColors[name])));
  g.appendChild(chip('ทั้งหมด', false, () => { resetGroups(); render(); }));
  g.appendChild(chip('ล้าง', false, () => { state.groups = new Set(); render(); }));

  document.getElementById('meta').textContent =
    `${V().items.length} ธีม · เทียบ ${benchLabel()} · ${DATA.asOf}`;
}

/* ── กราฟ ──────────────────────────────────────────── */
function renderPlot(){
  const items = visible();
  const W = 900, H = 660, M = {t:26, r:26, b:52, l:62};
  const host = document.getElementById('plot');

  if (!items.length){
    host.innerHTML = '<div style="padding:60px;text-align:center;color:var(--muted)">เลือกหมวดอย่างน้อยหนึ่งหมวดเพื่อดูแผนที่</div>';
    return;
  }

  const pts = [];
  items.forEach(it => it.points.slice(-state.tail).forEach(p => pts.push(p)));
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  let x0 = Math.min(100, ...xs), x1 = Math.max(100, ...xs);
  let y0 = Math.min(100, ...ys), y1 = Math.max(100, ...ys);
  const padX = Math.max((x1 - x0) * 0.12, 0.35), padY = Math.max((y1 - y0) * 0.12, 0.35);
  x0 -= padX; x1 += padX; y0 -= padY; y1 += padY;

  const X = v => M.l + (v - x0) / (x1 - x0) * (W - M.l - M.r);
  const Y = v => M.t + (1 - (v - y0) / (y1 - y0)) * (H - M.t - M.b);
  const cx = X(100), cy = Y(100);
  const f2 = v => v.toFixed(1);

  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="แผนภาพ RRG">
    <defs><filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="4" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter></defs>`;

  [['leading', cx, M.t, W - M.r - cx, cy - M.t],
   ['weakening', cx, cy, W - M.r - cx, H - M.b - cy],
   ['lagging', M.l, cy, cx - M.l, H - M.b - cy],
   ['improving', M.l, M.t, cx - M.l, cy - M.t]].forEach(([k, qx, qy, qw, qh]) => {
    if (qw > 0 && qh > 0)
      s += `<rect x="${qx}" y="${qy}" width="${qw}" height="${qh}" fill="${QUAD[k].raw}" opacity="0.045"/>`;
  });

  const ticks = (a, b) => { const o = [], st = (b - a) / 5;
    for (let i = 0; i <= 5; i++) o.push(a + st * i); return o; };
  ticks(x0, x1).forEach(t => {
    s += `<line x1="${X(t)}" y1="${M.t}" x2="${X(t)}" y2="${H - M.b}" stroke="rgba(255,255,255,.05)"/>
          <text x="${X(t)}" y="${H - M.b + 20}" fill="#8b8fb4" font-size="11" text-anchor="middle">${f2(t)}</text>`;
  });
  ticks(y0, y1).forEach(t => {
    s += `<line x1="${M.l}" y1="${Y(t)}" x2="${W - M.r}" y2="${Y(t)}" stroke="rgba(255,255,255,.05)"/>
          <text x="${M.l - 10}" y="${Y(t) + 4}" fill="#8b8fb4" font-size="11" text-anchor="end">${f2(t)}</text>`;
  });

  s += `<line x1="${cx}" y1="${M.t}" x2="${cx}" y2="${H - M.b}" stroke="rgba(255,255,255,.28)" stroke-dasharray="4 5"/>
        <line x1="${M.l}" y1="${cy}" x2="${W - M.r}" y2="${cy}" stroke="rgba(255,255,255,.28)" stroke-dasharray="4 5"/>`;

  [['improving', M.l + 12, M.t + 20, 'start'],
   ['leading', W - M.r - 12, M.t + 20, 'end'],
   ['lagging', M.l + 12, H - M.b - 12, 'start'],
   ['weakening', W - M.r - 12, H - M.b - 12, 'end']].forEach(([k, tx, ty, an]) => {
    s += `<text x="${tx}" y="${ty}" fill="${QUAD[k].raw}" font-size="13" font-weight="700"
           text-anchor="${an}" opacity=".85">${QUAD[k].th} · ${QUAD[k].en}</text>`;
  });

  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 12}" fill="#8b8fb4" font-size="12" text-anchor="middle">RS-Ratio → ความแข็งแกร่งเทียบ ${esc(benchLabel())}</text>
        <text x="16" y="${(M.t + H - M.b) / 2}" fill="#8b8fb4" font-size="12" text-anchor="middle"
          transform="rotate(-90 16 ${(M.t + H - M.b) / 2})">RS-Momentum → ทิศทางของความแข็งแกร่ง</text>`;

  items.forEach(it => {
    const tail = it.points.slice(-state.tail);
    if (!tail.length) return;
    const last = tail[tail.length - 1];
    if (tail.length > 1){
      const d = tail.map((p, i) => (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join(' ');
      s += `<path d="${d}" fill="none" stroke="${it.color}" stroke-width="2" stroke-linecap="round"
             stroke-linejoin="round" opacity=".55"/>`;
    }
    s += `<g><title>${esc(it.name)} · ${QUAD[it.quadrant].th}
RS-Ratio ${last[0].toFixed(2)} | RS-Momentum ${last[1].toFixed(2)}
${esc((it.members || []).join(' '))}</title>
          <circle cx="${X(last[0])}" cy="${Y(last[1])}" r="7" fill="${it.color}" filter="url(#glow)" opacity=".9"/>
          <circle cx="${X(last[0])}" cy="${Y(last[1])}" r="13" fill="${it.color}" opacity=".14"/></g>`;

    const show = state.labels === 'all' ||
                 (state.labels === 'leaders' && (it.quadrant === 'leading' || it.quadrant === 'improving'));
    if (show){
      const flip = X(last[0]) > W - M.r - 130;
      s += `<text x="${X(last[0]) + (flip ? -14 : 14)}" y="${Y(last[1]) + 4}" fill="#eef1ff"
             font-size="12.5" font-weight="600" text-anchor="${flip ? 'end' : 'start'}"
             stroke="#07060f" stroke-width="3.5" paint-order="stroke">${esc(it.name)}</text>`;
    }
  });

  host.innerHTML = s + '</svg>';
}

/* ── แผงด้านข้าง ────────────────────────────────────── */
function renderPanel(){
  const items = visible();
  const by = k => items.filter(i => i.quadrant === k).sort((a, b) => b.ratio - a.ratio);

  document.getElementById('counts').innerHTML =
    ['leading','weakening','lagging','improving'].map(k => `
      <div class="count" style="background:${QUAD[k].raw}12;border:1px solid ${QUAD[k].raw}33">
        <b style="color:${QUAD[k].raw}">${by(k).length}</b><span>${QUAD[k].th}</span>
      </div>`).join('');

  document.getElementById('lists').innerHTML =
    ['leading','improving','weakening','lagging'].map(k => {
      const list = by(k);
      const rows = list.length ? list.map(it => `
        <div class="row">
          <span class="dot" style="background:${it.color};box-shadow:0 0 8px ${it.color}"></span>
          <span class="nm" title="${esc((it.members || []).join(' '))}">${esc(it.name)}</span>
          <span class="val">${it.ratio.toFixed(1)} / ${it.momentum.toFixed(1)}</span>
        </div>`).join('')
        : `<div class="empty">${k === 'improving' ? 'ยังไม่มีธีมไหนเข้าโซนฟื้นตัว' : 'ไม่มีในโซนนี้'}</div>`;
      return `<section class="card sec"><h3 style="color:${QUAD[k].raw}">${QUAD[k].th} · ${QUAD[k].en}</h3>${rows}</section>`;
    }).join('');
}

/* ── Breadth: % หุ้นเหนือเส้นค่าเฉลี่ย ───────────────── */
const BR = DATA.breadth || null;
const ui = {tab: 'rrg', range: 250};
try { if (BR && localStorage.getItem('rrg-tab') === 'breadth') ui.tab = 'breadth'; } catch (e) {}
const TH_M = ['ม.ค.','ก.พ.','มี.ค.','เม.ย.','พ.ค.','มิ.ย.','ก.ค.','ส.ค.','ก.ย.','ต.ค.','พ.ย.','ธ.ค.'];
const fmtD = (iso, yr) => { const [y, m, d] = iso.split('-');
  return (+d) + ' ' + TH_M[+m - 1] + (yr ? ' ' + y.slice(2) : ''); };
function zone(p){
  if (p >= 80) return ['ร้อนแรง · กว้างเกือบทั้งตลาด', '#fbbf24'];
  if (p >= 60) return ['แข็งแรง · ส่วนใหญ่เป็นขาขึ้น', '#34d399'];
  if (p >= 40) return ['ก้ำกึ่ง', '#8b8fb4'];
  if (p >= 20) return ['อ่อนแอ · ส่วนใหญ่เป็นขาลง', '#fb7185'];
  return ['ล้างไพ่ · ต่ำสุดขั้ว', '#60a5fa'];
}

function renderTabs(){
  const host = document.getElementById('bar-tabs');
  host.innerHTML = '';
  if (!BR){ host.hidden = true; return; }
  lbl('หน้า', host);
  [['rrg', 'แผนที่ RRG'], ['breadth', 'Breadth · % เหนือ SMA' + BR.lenA]].forEach(([k, t]) =>
    host.appendChild(chip(t, ui.tab === k, () => {
      ui.tab = k;
      try { localStorage.setItem('rrg-tab', k); } catch (e) {}
      render();
    })));
  document.getElementById('tab-rrg').hidden = ui.tab !== 'rrg';
  document.getElementById('tab-breadth').hidden = ui.tab !== 'breadth';
}

function renderBreadth(){
  const bar = document.getElementById('bar-br');
  bar.innerHTML = '';
  lbl('ช่วงเวลา', bar);
  [[65, '3 เดือน'], [130, '6 เดือน'], [250, '1 ปี'], [0, 'ทั้งหมด']].forEach(([n, t]) =>
    bar.appendChild(chip(t, ui.range === n, () => { ui.range = n; render(); })));
  document.getElementById('meta').textContent =
    `ฐาน: ${BR.label} ${BR.total} ตัว · ${DATA.asOf}`;

  const N = BR.dates.length, s0 = ui.range ? Math.max(0, N - ui.range) : 0;
  const D = BR.dates.slice(s0), A = BR.pA.slice(s0), B = BR.pB.slice(s0), I = BR.index.slice(s0);
  const W = 900, H = 520, M = {t: 18, r: 66, b: 40, l: 50};
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const X = i => M.l + (D.length < 2 ? 0 : i / (D.length - 1) * iw);
  const Y = v => M.t + (1 - v / 100) * ih;
  const iv = I.filter(v => v != null), hasI = iv.length > 1;
  const lo = hasI ? Math.min(...iv) : 0, hi = hasI ? Math.max(...iv) : 1, pad = (hi - lo) * 0.06 || 1;
  const YI = v => M.t + (1 - (v - lo + pad) / (hi - lo + 2 * pad)) * ih;
  const path = (arr, f) => { let p = '', pen = false;
    arr.forEach((v, i) => { if (v == null){ pen = false; return; }
      p += (pen ? 'L' : 'M') + X(i).toFixed(1) + ',' + f(v).toFixed(1); pen = true; });
    return p; };

  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="กราฟ % หุ้นเหนือ SMA${BR.lenA}">`;
  g += `<rect x="${M.l}" y="${Y(100)}" width="${iw}" height="${Y(80) - Y(100)}" fill="#fbbf24" opacity=".07"/>`;
  g += `<rect x="${M.l}" y="${Y(20)}" width="${iw}" height="${Y(0) - Y(20)}" fill="#60a5fa" opacity=".08"/>`;
  [0, 20, 40, 50, 60, 80, 100].forEach(v => {
    g += `<line x1="${M.l}" x2="${M.l + iw}" y1="${Y(v)}" y2="${Y(v)}" stroke="rgba(255,255,255,${v === 50 ? .28 : .07})" ${v === 50 ? 'stroke-dasharray="5 5"' : ''}/>`;
    g += `<text x="${M.l - 10}" y="${Y(v) + 4}" text-anchor="end" fill="#8b8fb4" font-size="11" font-family="var(--mono)">${v}%</text>`;
  });
  for (let k = 0; hasI && k <= 4; k++){
    const v = lo - pad + (hi - lo + 2 * pad) * k / 4;
    g += `<text x="${M.l + iw + 10}" y="${YI(v) + 4}" fill="#8b8fb4" opacity=".7" font-size="11" font-family="var(--mono)">${v.toFixed(0)}</text>`;
  }
  const nt = Math.min(6, D.length);
  for (let k = 0; k < nt; k++){
    const i = Math.round(k * (D.length - 1) / Math.max(1, nt - 1));
    g += `<text x="${X(i)}" y="${H - 14}" text-anchor="middle" fill="#8b8fb4" font-size="11">${fmtD(D[i], true)}</text>`;
  }
  if (hasI) g += `<path d="${path(I, YI)}" fill="none" stroke="#eef1ff" stroke-opacity=".32" stroke-width="1.3"/>`;
  g += `<path d="${path(B, Y)}" fill="none" stroke="#a78bfa" stroke-opacity=".85" stroke-width="1.4"/>`;
  g += `<path d="${path(A, Y)}" fill="none" stroke="#67e8f9" stroke-width="2.4" style="filter:drop-shadow(0 0 5px #67e8f988)"/>`;
  const L = A.length - 1;
  g += `<circle cx="${X(L)}" cy="${Y(A[L])}" r="4.5" fill="#67e8f9"/>`;
  g += `<line id="br-x" y1="${M.t}" y2="${M.t + ih}" stroke="rgba(255,255,255,.35)" visibility="hidden"/>`;
  g += `<rect id="br-hit" style="touch-action:pan-y" x="${M.l}" y="${M.t}" width="${iw}" height="${ih}" fill="transparent"/></svg>`;

  const host = document.getElementById('br-plot');
  host.innerHTML =
    `<div class="legend"><span><i style="background:#67e8f9"></i>% เหนือ SMA${BR.lenA}</span>` +
    `<span><i style="background:#a78bfa"></i>% เหนือ SMA${BR.lenB}</span>` +
    `<span><i style="background:#eef1ff;opacity:.4"></i>${esc(benchLabel() || 'ดัชนี')} (แกนขวา)</span></div>` +
    g + '<div class="tip" id="br-tip"></div>';

  const svg = host.querySelector('svg'), tip = host.querySelector('#br-tip'), xl = host.querySelector('#br-x');
  const hit = host.querySelector('#br-hit');
  hit.addEventListener('pointermove', e => {
    const r = svg.getBoundingClientRect(), sx = (e.clientX - r.left) * W / r.width;
    const i = Math.max(0, Math.min(D.length - 1, Math.round((sx - M.l) / iw * (D.length - 1))));
    xl.setAttribute('x1', X(i)); xl.setAttribute('x2', X(i)); xl.setAttribute('visibility', 'visible');
    tip.innerHTML = `${fmtD(D[i], true)}<br><span style="color:#67e8f9">SMA${BR.lenA}</span> <b>${A[i] ?? '–'}%</b>` +
      ` · <span style="color:#a78bfa">SMA${BR.lenB}</span> <b>${B[i] ?? '–'}%</b><br>ดัชนี <b>${I[i] ?? '–'}</b>`;
    tip.style.display = 'block';
    const px = (X(i) / W) * r.width + 8 + (host.clientWidth - r.width) / 2;
    tip.style.left = Math.min(px, host.clientWidth - tip.offsetWidth - 8) + 'px';
    tip.style.top = '40px';
  });
  hit.addEventListener('pointerleave', () => { tip.style.display = 'none'; xl.setAttribute('visibility', 'hidden'); });

  // ── แผงข้าง ──
  const now = BR.pA[BR.pA.length - 1], nb = BR.pB[BR.pB.length - 1];
  const ago = n => BR.pA[Math.max(0, BR.pA.length - 1 - n)];
  const d5 = now - ago(5), d20 = now - ago(20);
  const [zt, zc] = zone(now);
  const sg = v => (v > 0 ? '+' : '') + v.toFixed(1);
  let h = `<div class="tiles">
    <div class="tile" style="border-color:${zc}55;background:${zc}12"><b style="color:${zc}">${now.toFixed(1)}%</b>
      <span>เหนือ SMA${BR.lenA} · ${BR.above}/${BR.total} ตัว</span></div>
    <div class="tile"><b style="color:#a78bfa">${nb.toFixed(1)}%</b><span>เหนือ SMA${BR.lenB}</span></div>
    <div class="tile"><b style="color:${d5 >= 0 ? '#34d399' : '#fb7185'}">${sg(d5)}</b><span>เปลี่ยน 5 แท่ง (จุด %)</span></div>
    <div class="tile"><b style="color:${d20 >= 0 ? '#34d399' : '#fb7185'}">${sg(d20)}</b><span>เปลี่ยน 20 แท่ง (จุด %)</span></div>
  </div>
  <div class="card sec"><h3 style="color:${zc}">${zt}</h3>
    <div class="empty">SMA${BR.lenB} ${nb > now ? 'สูงกว่า' : 'ต่ำกว่า'} SMA${BR.lenA} → แรงระยะสั้น${nb > now ? 'กำลังฟื้นก่อน' : 'อ่อนกว่าภาพใหญ่'}</div></div>`;

  h += `<div class="card sec"><h3 style="color:#67e8f9">รายหมวด · % เหนือ SMA${BR.lenA}</h3>` +
    BR.themes.map(t => { const c = zone(t.pct)[1];
      return `<div class="row" title="เหนือ: ${esc(t.above.join(', ') || '–')}&#10;ใต้: ${esc(t.below.join(', ') || '–')}">
        <span class="nm">${esc(t.name)}</span>
        <span class="meter"><i style="width:${t.pct}%;background:${c}"></i></span>
        <span class="val">${t.n}/${t.of}</span></div>`; }).join('') + '</div>';

  const list = (arr, c, t) => `<div class="card sec"><h3 style="color:${c}">${t} (${arr.length})</h3>` +
    (arr.length ? `<div class="tick">${arr.map(esc).join(' · ')}</div>` : '<div class="empty">ไม่มี</div>') + '</div>';
  h += list(BR.crossUp, '#34d399', `เพิ่งยืนเหนือ SMA${BR.lenA} ใน 5 แท่ง`);
  h += list(BR.crossDown, '#fb7185', `เพิ่งหลุด SMA${BR.lenA} ใน 5 แท่ง`);
  document.getElementById('br-side').innerHTML = h;

  document.getElementById('br-foot').innerHTML =
    `นับเฉพาะหุ้นที่มีข้อมูลครบพอคำนวณ SMA${BR.lenA} · ราคาปิดปรับเงินปันผล (Yahoo) · ` +
    `เกิน 80% = ตลาดร้อน มักใกล้พักตัว · ต่ำกว่า 20% = ล้างไพ่ มักใกล้จุดกลับตัว · ` +
    `ดัชนีขึ้นแต่เส้นนี้ลง = ขึ้นด้วยหุ้นไม่กี่ตัว (divergence)<br>ข้อมูลเพื่อการศึกษา ไม่ใช่คำแนะนำการลงทุน`;
}

function render(){
  renderTabs();
  if (ui.tab === 'breadth' && BR){ renderBreadth(); return; }
  renderBars(); renderPlot(); renderPanel();
}
render();
</script>
</body>
</html>
"""


def build_html(payload: dict, title: str, eyebrow: str, subtitle: str) -> str:
    return (HTML_TEMPLATE
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False))
            .replace("__TITLE__", title)
            .replace("__EYEBROW__", eyebrow)
            .replace("__SUBTITLE__", subtitle))


# ══════════════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="RRG หุ้นไทย → HTML dashboard")
    ap.add_argument("--benchmark", default="^SET.BK",
                    help="ดัชนีอ้างอิง: ^SET.BK ^SET50.BK ^SET100.BK ^MAI.BK "
                         "หรือ equal = ดัชนีถ่วงน้ำหนักเท่ากันจากหุ้นทุกตัวในธีม "
                         "(ตัดอคติจากหุ้นมาร์เก็ตแคปใหญ่)")
    ap.add_argument("--universe", default=None, choices=["pairs", "sector", "chain"],
                    help="ไม่ระบุ = สร้างครบทุกชุดไว้ในไฟล์เดียว สลับได้จากหน้าจอ | "
                         "pairs = สายสัมพันธ์ | sector = หมวดมาตรฐาน | chain = ห่วงโซ่มูลค่า")
    ap.add_argument("--min-members", type=int, default=2, dest="min_members",
                    help="ธีมต้องมีหุ้นที่ดึงข้อมูลได้อย่างน้อยกี่ตัวถึงจะแสดง (ค่าเริ่มต้น 2)")
    ap.add_argument("--open", action="store_true", help="เปิดไฟล์ HTML ในเบราว์เซอร์หลังรันเสร็จ")
    ap.add_argument("--min-turnover", type=float, default=0.0, dest="min_turnover",
                    help="กรองหุ้นที่มูลค่าซื้อขายเฉลี่ยต่อวันต่ำกว่ากี่ล้านบาท (0 = ไม่กรอง) "
                         "แนะนำ 20 สำหรับพอร์ตทั่วไป")
    ap.add_argument("--archive", action="store_true",
                    help="เก็บสำเนาลงโฟลเดอร์ history/ พร้อมวันที่ ไว้ย้อนดูการหมุนของธีม")
    ap.add_argument("--tickers", default="",
                    help="โหมดหุ้นรายตัว: คั่นด้วย comma เช่น PTT,KBANK,DELTA (ไม่ใส่ = โหมดธีม)")
    ap.add_argument("--period", default="2y", help="ช่วงข้อมูลที่ดึง (ค่าเริ่มต้น 2y)")
    ap.add_argument("--interval", default="1d", choices=["1d", "1wk"], help="ความถี่ข้อมูล")
    ap.add_argument("--window", type=int, default=60, help="หน้าต่าง z-score (แท่ง) ค่าเริ่มต้น 60")
    ap.add_argument("--mom", type=int, default=10, help="ช่วงคำนวณโมเมนตัม (แท่ง) ค่าเริ่มต้น 10")
    ap.add_argument("--smooth", type=int, default=3, help="EMA ปรับเส้นให้เรียบ ค่าเริ่มต้น 3")
    ap.add_argument("--tail", type=int, default=10, help="ความยาวหางเริ่มต้น (แท่ง)")
    ap.add_argument("--out", default=None,
                    help="ไฟล์ HTML ที่จะเขียน (ไม่ระบุ = เขียนไว้ข้าง ๆ ไฟล์สคริปต์)")
    ap.add_argument("--demo", action="store_true", help="ใช้ข้อมูลจำลอง ไม่ต้องต่อเน็ต")
    ap.add_argument("--breadth-list", default="", dest="breadth_list",
                    help="ไฟล์รายชื่อหุ้นสำหรับแท็บ Breadth (คั่นด้วย comma/ขึ้นบรรทัดใหม่) "
                         "ไม่ระบุ = ใช้หุ้นทุกตัวในแดชบอร์ด")
    args = ap.parse_args()

    auto_name = args.out is None
    single = bool(args.tickers.strip())

    # ── เลือกชุดธีมที่จะสร้าง (ไม่ระบุ = สร้างทุกชุดไว้ในไฟล์เดียว) ──────
    U_LABEL = {"pairs": "สายสัมพันธ์", "sector": "หมวดมาตรฐาน",
               "chain": "ห่วงโซ่มูลค่า", "watchlist": "หุ้นรายตัว"}
    if single:
        names = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        chosen = {"watchlist": {n: ("หุ้นรายตัว", [n]) for n in names}}
    elif args.universe:
        chosen = {args.universe: UNIVERSES[args.universe]}
    else:
        chosen = dict(UNIVERSES)

    all_syms = sorted({yahoo_symbol(t) for uni in chosen.values()
                       for _, members in uni.values() for t in members})
    index_sym = yahoo_symbol(args.benchmark)
    only_equal = args.benchmark.strip().lower() in ("equal", "ew", "เท่ากัน")

    # ── ฐานหุ้นสำหรับ Breadth ──────────────────────────────────────
    if args.breadth_list:
        raw_list = Path(args.breadth_list).read_text(encoding="utf-8")
        br_syms = [yahoo_symbol(t) for t in re.split(r"[,\s]+", raw_list) if t.strip()]
        br_label = Path(args.breadth_list).stem
    else:
        br_syms = [] if single else list(all_syms)
        br_label = "หุ้นทุกตัวในแดชบอร์ด"
    fetch_syms = sorted(set(all_syms) | set(br_syms))

    # ── ดึงข้อมูลรอบเดียว ใช้ร่วมกันทุกมุมมอง ──────────────────────────
    vol = None
    if args.demo:
        px = demo_prices(fetch_syms + [index_sym])
        px[index_sym] = px[all_syms].mean(axis=1)
    else:
        px, vol = fetch_prices(fetch_syms + [index_sym], args.period, args.interval)

    # ── Breadth (คำนวณก่อนกรองสภาพคล่อง ให้ฐานไม่เปลี่ยนตามตัวกรอง) ──────
    breadth = None
    if br_syms:
        n_long, n_short = (200, 50) if args.interval == "1d" else (40, 10)
        br_bench = px[index_sym] if index_sym in px.columns and not only_equal else None
        if br_bench is None or br_bench.tail(250).notna().mean() < 0.9:   # ดัชนีจาก Yahoo ขาดช่วง
            br_bench = composite_index(px[[c for c in br_syms if c in px.columns]])
        breadth = build_breadth(px, br_syms, br_bench, n_long, n_short, br_label)

    # ── ตัวกรองสภาพคล่อง ─────────────────────────────────────────────
    illiquid: dict[str, float] = {}
    if args.min_turnover > 0 and vol is not None:
        lookback = min(60, len(px))
        for c in all_syms:
            if c not in px.columns or c not in vol.columns:
                continue
            tv = (px[c] * vol[c]).tail(lookback).median()
            if pd.notna(tv) and tv < args.min_turnover * 1e6:
                illiquid[c] = float(tv) / 1e6
        if illiquid:
            px = px.drop(columns=list(illiquid))

    if index_sym in px.columns:
        s_ix = px[index_sym]
        print(f"benchmark {index_sym}: มีข้อมูล {int(s_ix.notna().sum())}/{len(s_ix)} แท่ง "
              f"ล่าสุด {s_ix.last_valid_index()}", file=sys.stderr)

    # ── benchmark ทั้งสองแบบ ─────────────────────────────────────────
    benches: dict[str, tuple[str, pd.Series]] = {}
    if not only_equal:
        if index_sym not in px.columns:
            sys.exit(f"ไม่พบข้อมูล benchmark {index_sym}")
        benches["index"] = (index_sym, px[index_sym].ffill().dropna())
    have_all = [c for c in all_syms if c in px.columns]
    benches["equal"] = ("ดัชนีถ่วงน้ำหนักเท่ากัน", composite_index(px[have_all]).dropna())

    # ── คำนวณทุกคู่ (ชุดธีม × benchmark) ─────────────────────────────
    views, missing, notes = {}, [], []
    for ukey, uni in chosen.items():
        for bkey, (blabel, bench) in benches.items():
            items, thin, dropped = [], [], []
            for i, (name, (group, members)) in enumerate(uni.items()):
                cols = [yahoo_symbol(m) for m in members
                        if yahoo_symbol(m) in px.columns
                        and px[yahoo_symbol(m)].notna().sum() > 30]
                gone = [m for m in members if yahoo_symbol(m) not in cols]
                missing += gone
                if len(cols) < args.min_members:
                    dropped.append(f"{name} (เหลือ {len(cols)}/{len(members)} ตัว)")
                    continue
                if gone:
                    thin.append(f"{name}: ใช้ {len(cols)}/{len(members)} — ขาด {', '.join(gone)}")
                co = rrg_coords(composite_index(px[cols]), bench,
                                args.window, args.mom, args.smooth)
                if co.empty:
                    continue
                tail = co.tail(30)
                last = tail.iloc[-1]
                items.append({
                    "name": name, "group": group,
                    "color": PALETTE[i % len(PALETTE)],
                    "ratio": round(float(last["ratio"]), 3),
                    "momentum": round(float(last["momentum"]), 3),
                    "quadrant": quadrant(float(last["ratio"]), float(last["momentum"])),
                    "members": [c.replace(".BK", "") for c in cols],
                    "points": [[round(float(r), 3), round(float(m), 3)]
                               for r, m in zip(tail["ratio"], tail["momentum"])],
                })
            if not items:
                continue
            groups = [g for g in GROUP_ORDER if any(it["group"] == g for it in items)]
            groups += sorted({it["group"] for it in items} - set(groups))
            views[f"{ukey}|{bkey}"] = {
                "groups": groups,
                "groupColors": {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)},
                "items": items,
            }
            if bkey == list(benches)[0]:
                notes.append((U_LABEL[ukey], thin, dropped, items))

    if not views:
        sys.exit("คำนวณไม่ได้ — ข้อมูลอาจสั้นเกินไป ลองเพิ่ม --period หรือลด --window")

    if auto_name:
        args.out = str(Path(__file__).resolve().parent
                       / f"rrg_thai_{args.interval}.html")

    as_of = px.index[-1]
    as_of_str = as_of.strftime("%d %b %Y") if hasattr(as_of, "strftime") else str(as_of)
    tf = "รายวัน" if args.interval == "1d" else "รายสัปดาห์"
    # ใช้เฉพาะชุด/benchmark ที่คำนวณได้จริง (กันหน้าเว็บว่างเมื่อ Yahoo ไม่มีข้อมูลดัชนี)
    live_b = [b for b in benches if any(k.endswith("|" + b) for k in views)]
    live_u = [u for u in chosen if any(k.startswith(u + "|") for k in views)]
    for b in benches:
        if b not in live_b:
            print(f"  ⚠ benchmark {benches[b][0]} คำนวณไม่ได้ (ข้อมูลไม่พอ) — ตัดออกจากหน้าเว็บ",
                  file=sys.stderr)
    default_u = live_u[0]
    default_b = live_b[0]

    payload = {
        "asOf": as_of_str,
        "defaultTail": args.tail,
        "tailOptions": sorted({5, args.tail, 20}),
        "universes": [{"key": k, "label": U_LABEL[k]} for k in live_u],
        "benchmarks": [{"key": k, "label": v[0].replace("^", "").replace(".BK", "")}
                       for k, v in benches.items() if k in live_b],
        "defaultUniverse": default_u,
        "defaultBenchmark": default_b,
        "views": views,
        "breadth": breadth,
    }

    title = "แผนที่ตามรอยเงิน — RRG หุ้นไทย" if not single else "RRG หุ้นรายตัว"
    eyebrow = f"SET · RRG {tf} · {len(views)} มุมมองในไฟล์เดียว"
    subtitle = (f"z-score window {args.window} แท่ง · โมเมนตัม {args.mom} แท่ง"
                + (f" · กรองสภาพคล่อง {args.min_turnover:.0f} ลบ./วัน" if args.min_turnover else "")
                + ("  ·  ⚠ ข้อมูลจำลอง (demo)" if args.demo else ""))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(build_html(payload, title, eyebrow, subtitle))

    print(f"\n✓ เขียนไฟล์ {args.out} ({len(views)} มุมมอง, ข้อมูลถึง {as_of_str})")
    for ulabel, thin, dropped, items in notes:
        summary = {}
        for it in items:
            summary.setdefault(it["quadrant"], []).append(it["name"])
        print(f"\n── {ulabel} ({len(items)} ธีม) ──")
        for k, th in [("leading", "นำตลาด"), ("improving", "กำลังฟื้น"),
                      ("weakening", "อ่อนแรง"), ("lagging", "ตามหลัง")]:
            n = summary.get(k, [])
            print(f"  {th:<10} {len(n):>2}  {', '.join(n) if n else '—'}")
        for t in thin:
            print(f"     ⚠ {t}")
        for t in dropped:
            print(f"     ✕ ตัดออก: {t}")

    if breadth:
        print(f"\n── Breadth ({breadth['label']}) ──\n  เหนือ SMA{breadth['lenA']}: "
              f"{breadth['pA'][-1]}% ({breadth['above']}/{breadth['total']} ตัว) · "
              f"เหนือ SMA{breadth['lenB']}: {breadth['pB'][-1]}%")
    elif br_syms:
        print("\n  ⚠ Breadth: ข้อมูลไม่พอ (ต้องการราคาย้อนหลังเกิน 200 แท่ง — ลอง --period 2y ขึ้นไป)")

    if illiquid:
        lst = ", ".join(f"{k.replace('.BK','')} ({v:.0f}ลบ.)"
                        for k, v in sorted(illiquid.items(), key=lambda x: x[1]))
        print(f"\n  ○ สภาพคล่องต่ำกว่า {args.min_turnover:.0f} ลบ./วัน: {lst}")
    if missing:
        print(f"\n  ⚠ ไม่พบข้อมูลบน Yahoo: {', '.join(sorted(set(missing)))}")

    if args.archive:
        stamp = (as_of.strftime("%Y-%m-%d") if hasattr(as_of, "strftime")
                 else datetime.now().strftime("%Y-%m-%d"))
        hist_dir = Path(args.out).resolve().parent / "history"
        hist_dir.mkdir(exist_ok=True)
        hist = hist_dir / f"{Path(args.out).stem}_{stamp}.html"
        hist.write_text(Path(args.out).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  ↳ เก็บสำเนาไว้ที่ {hist}")

    if args.open:
        import webbrowser
        webbrowser.open(Path(args.out).resolve().as_uri())


if __name__ == "__main__":
    main()
