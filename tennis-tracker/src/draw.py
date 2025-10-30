import cv2

import math

def draw_blob_label(frame, text, cx, top_y, fill,
                    text_color=(255, 255, 255),
                    font_scale=0.6, thickness=2,
                    margin_above=6, pad_x=12, pad_y=6):
    
    # text metrics
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    w = tw + pad_x * 2
    h = th + pad_y * 2
    r = h // 2  # end-cap radius

    # top-left of pill (place a bit above the given top_y)
    y1 = max(0, int(top_y) - h - margin_above)
    x1 = int(cx) - w // 2
    x2 = x1 + w
    y2 = y1 + h

    H, W = frame.shape[:2]
    # clamp horizontally (keep vertical placement)
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))

    # ---- draw filled pill with no seams (circles first, then a slightly
    #      oversized rectangle that overlaps them by 1px) ----
    cy = (y1 + y2) // 2
    # left cap
    cv2.circle(frame, (x1 + r, cy), r, fill, -1, lineType=cv2.LINE_AA)
    # right cap
    cv2.circle(frame, (x2 - r, cy), r, fill, -1, lineType=cv2.LINE_AA)
    # center (overlap +1 px to avoid any seam between shapes)
    cv2.rectangle(frame, (x1 + r - 1, y1), (x2 - r + 1, y2), fill, -1)

    # text centered
    tx = x1 + (w - tw) // 2
    ty = y1 + (h + th) // 2 - 2
    cv2.putText(frame, text, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
def draw_info_panel(frame, lines, corner="bl"):
   
    if not lines:
        return frame

    h, w = frame.shape[:2]
    pad_x = 25   # horizontal padding (was 20)
    pad_y = 16   # vertical padding (was 20)
    line_h = 32  # slightly taller line spacing
    panel_w = 500
    panel_h = pad_y * 2 + line_h * len(lines)

    if corner == "bl":
        x0, y0 = pad_x, h - panel_h - pad_y
    elif corner == "br":
        x0, y0 = w - panel_w - pad_x, h - panel_h - pad_y
    elif corner == "tr":
        x0, y0 = w - panel_w - pad_x, pad_y
    else:  # "tl"
        x0, y0 = pad_x, pad_y

    # --- background (solid dark gray) ---
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (60, 60, 60), -1)

    # --- white border ---
    cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (230, 230, 230), 2)

    # --- text (aligned nicely within bigger panel) ---
    y = y0 + pad_y + 25
    for txt in lines:
        cv2.putText(frame, txt, (x0 + pad_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y += line_h
    return frame

def draw_box_with_label(frame, x1, y1, x2, y2, color, title,
                        font_scale=0.5, font_thick=1, box_thick=2):
    x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color, box_thick)
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
    cv2.rectangle(frame, (x1i, max(0, y1i - th - 6)), (x1i + tw + 4, y1i), color, -1)
    cv2.putText(frame, title, (x1i + 2, y1i - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
