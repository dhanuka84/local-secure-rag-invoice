
import re, hashlib

def _vendor_guess(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    top = " ".join(lines[:5]).strip()
    m = re.search(r'([A-Z][A-Za-z0-9&\-\.\s]{2,})(?:\s*(Inc\.?|LLC|Ltd\.?))?', top)
    if m:
        return (m.group(0)).strip()
    return (lines[0][:40] if lines else "unknown").strip()

def _layout_hash(text: str) -> str:
    anchors = ["subtotal", "tax", "total"]
    order = []
    low = text.lower()
    for a in anchors:
        idx = low.find(a)
        order.append((a, idx if idx >= 0 else 999999))
    order = [a for a,_ in sorted(order, key=lambda x:x[1])]
    colish = "2col" if re.search(r":\s*\$?\d", low) else "1col"
    sig_basis = "|".join(order) + "|" + colish
    return hashlib.sha1(sig_basis.encode()).hexdigest()[:8]

def build_signature(text: str) -> str:
    vendor = re.sub(r'[^a-z0-9]+','_', _vendor_guess(text).lower()).strip('_')
    layout = _layout_hash(text)
    return f"{vendor}_{layout}"
