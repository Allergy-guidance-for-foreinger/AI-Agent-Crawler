"""학교 식단에서 독립된 안내 문구를 구분합니다."""

import re


def is_menu_notice(value: str) -> bool:
    """메뉴에 포함된 단어가 아닌, 항목 전체가 안내인 경우만 제외합니다."""
    text = " ".join(value.split()).strip()
    if not text or not any(char.isalnum() for char in text):
        return True
    # 강조용 별표는 음식명에도 사용되므로 별표만으로 제외하지 않습니다.
    text = text.strip("*★☆※· \t")
    plain = text.strip("[]() ")
    compact = re.sub(r"\s+", "", plain).casefold()
    if compact in {
        "끝", "종료", "end", "or", "또는", "미운영", "운영없음", "식당운영없음",
        "조식없음", "중식없음", "석식없음", "휴무", "휴관", "휴업",
        "추석", "추석연휴", "설날", "설연휴", "공휴일",
        "천원의아침밥", "천원의브런치", "1식4찬자율배식",
        "정식", "특식",
    }:
        return True
    if re.fullmatch(r"\d{1,2}:\d{2}\s*[~～\-]\s*\d{1,2}:\d{2}", plain):
        return True
    if re.fullmatch(r"천원의\s*(?:아침밥|브런치)\s*\(.+에서\s*운영\)", plain):
        return True
    if re.fullmatch(r"(?:풍성한|즐거운|행복한)\s*(?:한가위|추석|설날).*(?:되세요|보내세요)[\s^!~.♥♡]*", plain):
        return True
    if re.fullmatch(r".+(?:식당|관)에서(?:\s*운영)?", plain):
        return True
    if re.fullmatch(r"위의\s*정식메뉴\s*운영", plain):
        return True
    if re.fullmatch(r"재학생만\s*해당", plain):
        return True
    return False
