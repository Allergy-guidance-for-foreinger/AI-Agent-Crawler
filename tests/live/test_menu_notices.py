from datetime import date

import pytest

from app.domain.crawler.menu_filter import is_menu_notice
from app.domain.crawler.gknu_menu import parse_gknu_day_html
from app.domain.crawler.knu_menu import parse_knu_week_html
from app.services.ops import parse_menu_cell, _expand_bunsik_category_tokens


@pytest.mark.parametrize("text", [
    "------", "─ ─ ─", "끝", " 끝 ", "*추석연휴*", "미 운 영",
    "석식없음", "(11:00~11:40)", "일품식당에서", "위의 정식메뉴 운영",
    "풍성한 한가위 되세요 ^^", "[천원의 아침밥 (채움관에서 운영)]", "or",
    "천원의 아침밥", "1식4찬 자율배식",
])
def test_notice_is_rejected(text):
    assert is_menu_notice(text)


@pytest.mark.parametrize("text", [
    "*돼지국밥", "고추&양파*쌈장", "불고기비빔밥★", "끝내주는돈까스",
    "추석송편", "orzo 파스타", "일품정식(석식)", "배추김치/요구르트",
])
def test_actual_menu_is_preserved(text):
    assert not is_menu_notice(text)


def test_kumoh_notices_do_not_become_menu_or_corner():
    cell = "------|||중식|||*돼지국밥|||깍두기|||끝|||(11:00~11:40)|||일품식당에서|||위의 정식메뉴 운영|||"
    assert parse_menu_cell(cell, "정찬식당") == (
        "중식", "LUNCH", ["*돼지국밥", "깍두기"]
    )
    assert parse_menu_cell("중식|||추석연휴|||", "정찬식당")[2] == []


def test_highlighted_bunsik_category_uses_existing_expansion():
    assert _expand_bunsik_category_tokens(['*돈가스류']) == [
        '왕돈가스', '고구마돈가스', '치즈돈가스'
    ]


@pytest.mark.parametrize("body", [
    "<ul class='menu_im'><li>끝</li><li>풍성한 한가위 되세요 ^^</li><li>쌀밥</li></ul>",
    "끝￦ 1,000쌀밥￦ 2,000",
])
def test_knu_notice_filter_applies_to_both_html_layouts(body):
    html = f"<table class='tstyle_me'><tr><th>구분</th><th>월<p class='week_t'>(09/21)</p></th></tr></table><div class='week_table'>중식<table><tr><td>{body}</td></tr></table></div>"
    meals = parse_knu_week_html(html, cafeteria_name="정보센터식당", start=date(2026,9,21), end=date(2026,9,21))
    assert [m['menuName'] for meal in meals for m in meal['menus']] == ['쌀밥']
    assert meals[0]['menus'][0]['displayOrder'] == 1


def test_gknu_removes_notices_and_empty_meal_slots():
    html = '<dl><dt>조식</dt><dd>[천원의 아침밥]<br>흑미밥<br>or<br>왕김밥<br>끝</dd></dl><dl><dt>석식</dt><dd>미운영</dd></dl>'
    meals = parse_gknu_day_html(html, meal_date=date(2026, 9, 21))
    assert len(meals) == 1
    assert [m['menuName'] for m in meals[0]['menus']] == ['흑미밥', '왕김밥']
    assert [m['displayOrder'] for m in meals[0]['menus']] == [1, 2]
