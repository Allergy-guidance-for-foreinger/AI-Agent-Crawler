#!/usr/bin/env python3
"""음식 이미지를 Gemini Vision으로 분석합니다."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import repo_env
from google import genai
from google.genai import types

from utils.json_extract import extract_json_object

_IMAGE_JSON_SCHEMA = """{
  "음식명": "이미지에서 보이는 음식 이름(한국어)",
  "추정_식재료": [
    {"재료": "주요 식재료(한국어)", "근거": "이미지에서 보이는 근거"}
  ]
}"""


def _build_image_analysis_prompt(*, menu_name: str | None) -> str:
    hint = ""
    if menu_name:
        hint = f"\n참고: 업로드한 메뉴명은 「{menu_name}」입니다. 이미지와 일치하면 음식명에 반영하세요."
    return f"""당신은 음식 사진을 보고 식재료를 추정하는 도우미입니다.
반드시 한국어로 답하고, 아래 JSON 객체 형식 하나만 출력하세요.{hint}

{_IMAGE_JSON_SCHEMA}

규칙:
- 추정_식재료: 눈에 보이거나 이 음식에 흔히 들어가는 주재료를 3개 이상 나열하세요.
- 키 이름은 위와 동일하게 「음식명」「추정_식재료」「재료」「근거」만 사용하세요.
- 재료를 알 수 없으면 추정_식재료를 빈 배열 []로 두세요.
"""


def extract_ingredient_names_from_image_analysis(analysis: dict[str, Any]) -> list[str]:
    """Gemini 이미지 분석 JSON에서 재료명 목록을 추출합니다 (키 형식 여러 가지 허용)."""
    names: list[str] = []

    def add_name(raw: Any) -> None:
        s = str(raw).strip()
        if s and s not in names:
            names.append(s)

    candidates: list[Any] = []
    for key in ("추정_식재료", "ingredientsKo", "ingredients", "식재료"):
        value = analysis.get(key)
        if isinstance(value, list) and value:
            candidates = value
            break

    for item in candidates:
        if isinstance(item, dict):
            for field in ("재료", "name", "ingredient", "ingredientName"):
                if item.get(field):
                    add_name(item[field])
                    break
        elif isinstance(item, str):
            add_name(item)

    return names


def extract_food_name_from_image_analysis(analysis: dict[str, Any]) -> str | None:
    """Gemini 이미지 분석 JSON에서 음식명을 추출합니다."""
    for key in ("음식명", "foodNameKo", "foodName"):
        value = analysis.get(key)
        if isinstance(value, str):
            name = value.strip()
            if name:
                return name
    return None


def _guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "image/jpeg"


def analyze_food_image_bytes(
    client: genai.Client,
    model_name: str,
    image_bytes: bytes,
    mime_type: str,
    menu_name: str | None = None,
) -> dict[str, Any]:
    prompt = _build_image_analysis_prompt(menu_name=menu_name)
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=2048,
            response_mime_type="application/json",
        ),
    )
    raw = (response.text or "").strip()
    if not raw:
        raise RuntimeError("모델 응답이 비어 있습니다.")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = extract_json_object(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("모델 응답 JSON이 객체 형태가 아닙니다.")
    return parsed


def analyze_food_image(
    client: genai.Client,
    model_name: str,
    image_path: str | Path,
    menu_name: str | None = None,
) -> dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")
    image_bytes = path.read_bytes()
    mime_type = _guess_mime_type(path)
    return analyze_food_image_bytes(client, model_name, image_bytes, mime_type, menu_name)


def main() -> None:
    repo_env.load_dotenv_from_repo_root()
    parser = argparse.ArgumentParser(description="음식 이미지 식재료 추정 (Gemini)")
    parser.add_argument("image", nargs="?", default="test_image.jpeg")
    parser.add_argument("--menu-name", default="", help="참고 메뉴명")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    args = parser.parse_args()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY가 없습니다.")
    client = genai.Client(api_key=api_key)
    menu_name = (args.menu_name or "").strip() or None
    result = analyze_food_image(client, args.model, args.image, menu_name)
    names = extract_ingredient_names_from_image_analysis(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n추출 재료({len(names)}): {names}")


if __name__ == "__main__":
    main()
