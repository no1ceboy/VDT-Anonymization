"""Deterministic synthetic vocabulary. No detection or linking rules.

Expanded using selected examples from the local VLSP2016 training vocabulary
(datnth1709/VLSP2016-NER-data; numeric PER/ORG mapping inferred from examples).
Selections are editorial: corpus annotations do not supply gender or given-name
boundaries. Rare fragments and foreign names were not imported automatically.
"""

ORGANIZATION_NAME_LEFT = [
    "An", "Bình", "Cao", "Đại", "Đông", "Gia", "Hải", "Hòa", "Hưng", "Kim",
    "Long", "Minh", "Nam", "Phú", "Quang", "Sơn", "Tân", "Thái", "Thanh",
    "Thiên", "Thịnh", "Trường", "Việt", "Vĩnh", "Xuân",
    # Components observed in Hoàng Đạt, Khai Minh, Hùng Vương, Hồng Lĩnh,
    # Trung Kiên, Triều Phú, Phong Phú and Phương Nam.
    "Hoàng", "Khai", "Hùng", "Hồng", "Trung", "Triều", "Phong", "Phương",
]
ORGANIZATION_NAME_RIGHT = [
    "An", "Bình", "Châu", "Đức", "Gia", "Hải", "Hòa", "Khang", "Long", "Minh",
    "Phát", "Phú", "Quang", "Sơn", "Tâm", "Thành", "Thịnh", "Tiến", "Trung",
    "Việt", "Vinh", "Yên", "Nguyên", "Đạt", "Lộc",
    "Vương", "Lĩnh", "Kiên", "Hữu", "Đồng",
]
SYNTHETIC_ORGANIZATION_NAMES = list(dict.fromkeys([
    "Minh Việt", "Đại Việt", "An Phát", "Hưng Thịnh", "Tân Thành", "Phú Gia",
    "Thịnh Vượng", "Thanh Bình", "Hoàng Gia", "Việt An",
    *[
    f"{left} {right}"
    for left in ORGANIZATION_NAME_LEFT
    for right in ORGANIZATION_NAME_RIGHT
    if left != right
    ],
]))

# These are given-name pools, not recovered identities. They contain only
# common, single-token Vietnamese given names. We preserve the anonymized
# initial when a suitable common name exists, but do not force an uncommon or
# invented name just to match an arbitrary publication marker.
GIVEN_NAMES = {
    "male": {
        "A": ["An", "Anh"],
        "B": ["Bảo", "Bình"],
        "C": ["Cường", "Chính", "Công", "Chí", "Chương", "Chiến", "Cảnh"],
        "D": ["Dũng", "Duy", "Dương"],
        "Đ": ["Đạt", "Đức", "Đăng", "Đại", "Đông", "Định"],
        "G": ["Gia", "Giang"],
        "H": ["Hải", "Hùng", "Hoàng", "Hưng", "Hiếu", "Huy", "Hào", "Hiệp", "Hiển", "Huấn"],
        "K": ["Khoa", "Kiên", "Khôi", "Khánh", "Khang", "Khải", "Khương", "Kha"],
        "L": ["Long", "Lâm", "Lộc", "Luân", "Lân", "Liêm", "Lợi"],
        "M": ["Minh", "Mạnh"],
        "N": ["Nam", "Nghĩa", "Nhân", "Nguyên", "Ngọc", "Nhật"],
        "P": ["Phúc", "Phong", "Phát", "Phước"],
        "Q": ["Quang", "Quốc", "Quân", "Quí", "Quyền"],
        "S": ["Sơn", "Sang", "Sinh"],
        "T": ["Tuấn", "Tùng", "Trung", "Thành", "Thắng", "Tiến", "Tâm", "Thịnh", "Trí",
              "Tấn", "Toàn", "Tân", "Tài", "Thiện", "Thái", "Thông", "Trường", "Thuận", "Triết"],
        "V": ["Việt", "Vinh", "Vũ", "Vương"],
        "X": ["Xuân"],
    },
    "female": {
        "A": ["Ánh", "An", "Anh"],
        "B": ["Bích", "Bình", "Bảo"],
        "C": ["Chi", "Châu", "Cúc"],
        "D": ["Diễm", "Dung", "Duyên", "Diệp"],
        "Đ": ["Đan", "Đào"],
        "G": ["Giang", "Giao"],
        "H": ["Hạnh", "Hoa", "Hương", "Hiền", "Hồng", "Hà", "Hằng", "Huyền", "Hoài", "Huệ"],
        "K": ["Kim", "Kiều", "Khánh", "Khuê"],
        "L": ["Lan", "Linh", "Loan", "Liên"],
        "M": ["Mai", "My", "Mỹ"],
        "N": ["Ngân", "Nga", "Ngọc", "Nhung", "Nhi", "Như", "Nguyên", "Nhã"],
        "P": ["Phương", "Phượng"],
        "Q": ["Quỳnh", "Quyên"],
        "S": ["Sen", "Sương"],
        "T": ["Trang", "Thảo", "Trinh", "Thúy", "Thanh", "Tâm", "Tiên", "Tuyết", "Tú",
              "Trâm", "Thư", "Trúc", "Thu", "Thoa", "Thắm", "Trân", "Thương"],
        "V": ["Vân", "Vy", "Vi"],
        "X": ["Xuân"],
        "Y": ["Yến"],
        "U": ["Uyên"],
        "O": ["Oanh"],
    },
}

FAMILY_NAMES = [
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Võ", "Vũ", "Đặng", "Bùi",
    "Phan", "Đoàn", "Đinh", "Dương", "Hồ", "Ngô", "Đỗ", "Tô", "Cao", "Lý", "Lưu",
    "Trương", "Mai", "Mạc", "Hà", "Giang", "Chung", "Chu", "Thái", "Vương", "Lương",
]

FALLBACK_GIVEN_NAMES = {
    "male": ["Anh", "Bình", "Dũng", "Hải", "Hùng", "Khoa", "Long", "Minh", "Nam", "Phúc", "Quang", "Sơn", "Tuấn", "Vinh"],
    "female": ["Anh", "Bình", "Dung", "Hạnh", "Hoa", "Hương", "Lan", "Linh", "Mai", "Nga", "Ngọc", "Phương", "Trang", "Thảo", "Vy"],
}

# Only these tokens may be used to expand a one-letter marker into a full
# name. This prevents words such as ``Bên``, ``Cháu`` or all-caps headings from
# becoming fake name prefixes when the marker regex scans the whole document.
NAME_PREFIX_TOKENS = {
    "nguyễn", "trần", "lê", "phạm", "hoàng", "huỳnh", "võ", "vũ", "đặng", "bùi",
    "phan", "đoàn", "đinh", "dương", "hồ", "ngô", "đỗ", "tô", "tạ", "cao", "mai",
    "lý", "lưu", "trương", "mạc", "mã", "hứa", "giàng", "chung", "chu", "hà", "hạ",
    "lâm", "tống", "tôn", "kiều", "khổng", "thân", "bạch", "vi", "thái", "quách",
    "triệu", "tăng", "từ", "nông", "sầm", "thạch", "vương", "lương", "uông", "nghiêm",
    "lại", "chế", "văn", "thị", "đức", "hữu", "quốc", "ngọc", "minh", "thanh", "đình",
    "công", "thế", "xuân", "kim", "thúy", "nhật", "quang", "tấn", "thành", "anh", "bảo",
    "gia", "khắc", "trọng", "phước", "phú", "cẩm", "hải", "mạnh", "đình", "văn",
    "trúc", "bích", "bạch", "hoài", "thùy", "tuệ", "thái", "khắc", "tú", "tường",
    "huyền", "diệp", "phương", "phượng", "quỳnh", "quyên", "lan", "linh", "loan", "hùng",
}

# Plausible synthetic place names by administrative unit. These names are
# intentionally generic and are selected deterministically by marker/index.
LOCATION_NAMES = {
    "province": {
        "A": ["An Bình"], "B": ["Bình An"], "C": ["Cao Sơn"],
        "D": ["Đông Hải"], "Đ": ["Đông Hải"], "H": ["Hòa Bình"],
        "L": ["Lâm An"], "M": ["Minh Hải"], "N": ["Nam Sơn"],
        "P": ["Phú An"], "Q": ["Quang Minh"], "S": ["Sơn Hà"],
        "T": ["Tân Bình"], "V": ["Vĩnh An"],
    },
    "district": {
        "A": ["An Sơn"], "B": ["Bình Sơn"], "C": ["Cao Sơn"],
        "D": ["Đông Sơn"], "Đ": ["Đức Sơn"], "H": ["Hòa Sơn"],
        "L": ["Lộc Bình"], "M": ["Minh Sơn"], "N": ["Nam Sơn"],
        "P": ["Phú Sơn"], "Q": ["Quang Sơn"], "S": ["Sơn Tây"],
        "T": ["Tân Sơn"], "V": ["Vĩnh Sơn"],
    },
    "commune": {
        "A": ["An Hòa"], "B": ["Bình Minh"], "C": ["Cao An"],
        "D": ["Đông An"], "Đ": ["Đức An"], "H": ["Hòa An"],
        "L": ["Long An"], "M": ["Minh An"], "N": ["Nam An"],
        "P": ["Phú An"], "Q": ["Quang An"], "S": ["Sơn An"],
        "T": ["Tân An"], "V": ["Vĩnh An"],
    },
    "hamlet": {
        "A": ["An Bình"], "B": ["Bình An"], "C": ["Cầu Mới"],
        "D": ["Đông Bình"], "Đ": ["Đức Hòa"], "H": ["Hòa Bình"],
        "L": ["Long Bình"], "M": ["Minh Tân"], "N": ["Nam Bình"],
        "P": ["Phú Bình"], "Q": ["Quang Trung"], "S": ["Sơn Bình"],
        "T": ["Tân Bình"], "V": ["Vĩnh Bình"],
    },
    "generic": {
        "A": ["An Bình"], "B": ["Bình Minh"], "C": ["Cao Sơn"],
        "D": ["Đông An"], "Đ": ["Đức An"], "H": ["Hòa Bình"],
        "L": ["Long An"], "M": ["Minh An"], "N": ["Nam Sơn"],
        "P": ["Phú An"], "Q": ["Quang Minh"], "S": ["Sơn Hà"],
        "T": ["Tân Bình"], "V": ["Vĩnh An"],
    },
}
