#!/usr/bin/env python3
"""다량 메뉴명으로 메뉴 분석(Gemini) 일괄 실테스트."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import repo_env

repo_env.load_dotenv_from_repo_root()

from app.config.runtime import load_runtime_context
from app.services.menu_analysis_builder import (
    build_menu_analysis_failed_result,
    build_menu_analysis_success_result,
)
from app.services.ops import analyze_food_text

# 급식·학식에서 흔한 메뉴명 + 엣지 케이스
MENU_NAMES = [
    "김치찌개",
    "된장찌개",
    "순두부찌개",
    "부대찌개",
    "미역국",
    "콩나물국",
    "불고기",
    "제육덮밥",
    "돈까스",
    "치킨가라아게",
    "닭갈비",
    "오징어볶음",
    "고등어구이",
    "계란찜",
    "비빔밥",
    "잔치국수",
    "카레라이스",
    "스파게티",
    "짜장면",
    "짬뽕",
    "떡볶이",
    "라볶이",
    "김밥",
    "참치김밥",
    "라면",
    "우동",
    "쫄면",
    "갈비탕",
    "설렁탕",
    "삼겹살구이",
    "햄버거스테이크",
    "샐러드",
    "토마토파스타",
    "크림스프",
    "팽이버섯볶음밥",
    "낙지비빔밥",
    "해물파전",
    "김치전",
    "치즈돈까스",
    "마라탕",
    "탕수육",
    "깐풍기",
    "유부초밥",
    "연어덮밥",
    "새우튀김우동",
    "돼지고기김치찌개",
    "김치찌개/공기밥",
    "제육+계란후라이",
    "치즈불닭볶음밥",
    "고추장삼겹살볶음",
]


def main() -> None:
    ctx = load_runtime_context()
    if not ctx.client:
        raise SystemExit("GEMINI_API_KEY 없음")

    model = ctx.config.gemini_model
    tz = ZoneInfo(ctx.config.timezone_name)
    print(f"model={model} menus={len(MENU_NAMES)}\n")

    rows: list[dict] = []
    failures = 0
    t0 = time.perf_counter()

    for i, name in enumerate(MENU_NAMES, start=1):
        analyzed_at = datetime.now(tz).isoformat(timespec="seconds")
        try:
            raw = analyze_food_text(ctx.client, model, name)
        except (RuntimeError, JSONDecodeError) as e:
            failures += 1
            result = build_menu_analysis_failed_result(
                menu_id=i,
                menu_name=name,
                model_name="gemini",
                model_version=model,
                analyzed_at=analyzed_at,
                reason=str(e),
            )
        else:
            result = build_menu_analysis_success_result(
                menu_id=i,
                menu_name=name,
                model_name="gemini",
                model_version=model,
                analyzed_at=analyzed_at,
                analysis=raw,
            )

        ings = result.get("ingredients") or []
        algs = result.get("allergies") or []
        unmapped = result.get("unmappedAllergenNames") or []
        coded_ings = [x for x in ings if x.get("ingredientCode")]
        rows.append(
            {
                "menuName": name,
                "status": result.get("status"),
                "ingredientCount": len(ings),
                "ingredientWithCode": len(coded_ings),
                "allergyCount": len(algs),
                "allergyCodes": [a.get("allergyCode") for a in algs],
                "unmappedAllergenNames": unmapped,
                "spicyLevel": result.get("spicyLevel"),
                "ingredientNames": [x.get("ingredientName") for x in ings],
                "reason": result.get("reason"),
            }
        )
        mark = "OK" if result.get("status") == "SUCCESS" else "FAIL"
        print(
            f"[{i:02d}/{len(MENU_NAMES)}] {mark} {name} | "
            f"재료 {len(ings)} (코드 {len(coded_ings)}) | "
            f"알레르기 {len(algs)} {[a.get('allergyCode') for a in algs]} | "
            f"미매핑 {unmapped} | 매움 {result.get('spicyLevel')}"
        )
        time.sleep(0.5)

    elapsed = time.perf_counter() - t0
    success = sum(1 for r in rows if r["status"] == "SUCCESS")
    total_ings = sum(r["ingredientCount"] for r in rows)
    total_coded = sum(r["ingredientWithCode"] for r in rows)
    total_alg = sum(r["allergyCount"] for r in rows)
    any_unmapped = sum(1 for r in rows if r["unmappedAllergenNames"])

    print("\n=== 요약 ===")
    print(f"성공 {success}/{len(MENU_NAMES)}, 실패 {failures}")
    print(f"총 재료 항목 {total_ings}, 코드 있음 {total_coded} ({100*total_coded/max(total_ings,1):.1f}%)")
    print(f"총 알레르기 항목 {total_alg}, 미매핑 알레르기명 있는 메뉴 {any_unmapped}건")
    print(f"소요 {elapsed:.1f}s")

    out = ROOT / "scripts" / "batch_menu_analysis_test_results.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n상세 JSON: {out}")


if __name__ == "__main__":
    main()
