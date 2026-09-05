# Spring 식단 크롤 API — 실제 요청/응답 예시

Spring 연동 확인용으로, **실제로 호출한 요청·응답**을 기록한 문서입니다.

## 공통

| 항목 | 값 |
|---|---|
| Wrapped 엔드포인트 | `POST /api/v1/python/meals/crawl` |
| Unwrapped (Spring) | `POST /api/v1/crawl/meals` — 아래 응답의 `data` 본문만 반환 |
| 조회 기간 | `2026-09-05` ~ `2026-09-07` |
| Accept-Language | `ko` |

> **경북대·경국대**: DB에 학교 URL만 있어도 됩니다. `sourceUrl`에 `https://coop.knu.ac.kr` / `https://www.gknu.ac.kr`만 넣고
> `cafeteriaName`으로 식당을 구분하면, Python 서버가 식당별 URL(`shop_sqno` / `menu_idx`)로 보정합니다.
>
> **금오**: 식당마다 URL이 달라서 기존처럼 식당별 `sourceUrl`을 보냅니다.

## 요약

| 학교 | 식당 | sourceUrl | HTTP | meals 수 |
|---|---|---|---|---|
| 금오공과대학교 | 일품식당 | `https://www.kumoh.ac.kr/ko/restaurant01.do` | 200 | 4 |
| 금오공과대학교 | 정찬식당 | `https://www.kumoh.ac.kr/ko/restaurant02.do` | 200 | 2 |
| 금오공과대학교 | 분식당 | `https://www.kumoh.ac.kr/ko/restaurant04.do` | 200 | 4 |
| 경북대학교 | 정보센터식당 | `https://coop.knu.ac.kr` | 200 | 6 |
| 경북대학교 | 복지관 교직원식당 | `https://coop.knu.ac.kr` | 200 | 3 |
| 경북대학교 | 카페테리아 첨성 | `https://coop.knu.ac.kr` | 200 | 6 |
| 경북대학교 | GP감꽃식당 | `https://coop.knu.ac.kr` | 200 | 3 |
| 경북대학교 | 공학관교직원식당(외부업체) | `https://coop.knu.ac.kr` | 200 | 6 |
| 국립경국대학교 | 이룸관(안동, 학생식당) | `https://www.gknu.ac.kr` | 200 | 2 |
| 국립경국대학교 | 채움관(안동, 교직원식당) | `https://www.gknu.ac.kr` | 200 | 2 |
| 국립경국대학교 | 양식코너(안동) | `https://www.gknu.ac.kr` | 200 | 1 |
| 국립경국대학교 | 학생식당(예천) | `https://www.gknu.ac.kr` | 200 | 7 |

---

## 금오공과대학교

<details>
<summary><strong>일품식당</strong> — HTTP 200, meals=4 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "금오공과대학교",
  "cafeteriaName": "일품식당",
  "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant01.do",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "금오공과대학교",
    "cafeteriaName": "일품식당",
    "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant01.do",
    "startDate": "2026-09-05",
    "endDate": "2026-09-07",
    "meals": [
      {
        "mealDate": "2026-09-05",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "미운영"
          }
        ]
      },
      {
        "mealDate": "2026-09-05",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "미운영"
          }
        ]
      },
      {
        "mealDate": "2026-09-06",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "미운영"
          }
        ]
      },
      {
        "mealDate": "2026-09-06",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "미운영"
          }
        ]
      }
    ]
  }
}
```

<details>
<summary>Unwrapped (`POST /api/v1/crawl/meals`) 형태 — data 본문만</summary>

```json
{
  "schoolName": "금오공과대학교",
  "cafeteriaName": "일품식당",
  "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant01.do",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07",
  "meals": [
    {
      "mealDate": "2026-09-05",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "미운영"
        }
      ]
    },
    {
      "mealDate": "2026-09-05",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "미운영"
        }
      ]
    },
    {
      "mealDate": "2026-09-06",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "미운영"
        }
      ]
    },
    {
      "mealDate": "2026-09-06",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "미운영"
        }
      ]
    }
  ]
}
```

</details>

</details>

<details>
<summary><strong>정찬식당</strong> — HTTP 200, meals=2 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "금오공과대학교",
  "cafeteriaName": "정찬식당",
  "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant02.do",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "금오공과대학교",
    "cafeteriaName": "정찬식당",
    "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant02.do",
    "startDate": "2026-09-05",
    "endDate": "2026-09-07",
    "meals": [
      {
        "mealDate": "2026-09-05",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "중식",
            "displayOrder": 1,
            "menuName": "미운영"
          }
        ]
      },
      {
        "mealDate": "2026-09-06",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "중식",
            "displayOrder": 1,
            "menuName": "미운영"
          }
        ]
      }
    ]
  }
}
```

<details>
<summary>Unwrapped (`POST /api/v1/crawl/meals`) 형태 — data 본문만</summary>

```json
{
  "schoolName": "금오공과대학교",
  "cafeteriaName": "정찬식당",
  "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant02.do",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07",
  "meals": [
    {
      "mealDate": "2026-09-05",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "중식",
          "displayOrder": 1,
          "menuName": "미운영"
        }
      ]
    },
    {
      "mealDate": "2026-09-06",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "중식",
          "displayOrder": 1,
          "menuName": "미운영"
        }
      ]
    }
  ]
}
```

</details>

</details>

<details>
<summary><strong>분식당</strong> — HTTP 200, meals=4 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "금오공과대학교",
  "cafeteriaName": "분식당",
  "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant04.do",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "금오공과대학교",
    "cafeteriaName": "분식당",
    "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant04.do",
    "startDate": "2026-09-05",
    "endDate": "2026-09-07",
    "meals": [
      {
        "mealDate": "2026-09-05",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "미운영"
          }
        ]
      },
      {
        "mealDate": "2026-09-05",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "미운영"
          }
        ]
      },
      {
        "mealDate": "2026-09-06",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "미운영"
          }
        ]
      },
      {
        "mealDate": "2026-09-06",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "미운영"
          }
        ]
      }
    ]
  }
}
```

<details>
<summary>Unwrapped (`POST /api/v1/crawl/meals`) 형태 — data 본문만</summary>

```json
{
  "schoolName": "금오공과대학교",
  "cafeteriaName": "분식당",
  "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant04.do",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07",
  "meals": [
    {
      "mealDate": "2026-09-05",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "미운영"
        }
      ]
    },
    {
      "mealDate": "2026-09-05",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "미운영"
        }
      ]
    },
    {
      "mealDate": "2026-09-06",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "미운영"
        }
      ]
    },
    {
      "mealDate": "2026-09-06",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "미운영"
        }
      ]
    }
  ]
}
```

</details>

</details>

---

## 경북대학교

<details>
<summary><strong>정보센터식당</strong> — HTTP 200, meals=6 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "정보센터식당",
  "sourceUrl": "https://coop.knu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "경북대학교",
    "cafeteriaName": "정보센터식당",
    "sourceUrl": "https://coop.knu.ac.kr",
    "startDate": "2026-09-05",
    "endDate": "2026-09-07",
    "meals": [
      {
        "mealDate": "2026-09-05",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "특식"
          }
        ]
      },
      {
        "mealDate": "2026-09-05",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "특식"
          }
        ]
      },
      {
        "mealDate": "2026-09-06",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "특식"
          }
        ]
      },
      {
        "mealDate": "2026-09-06",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "특식"
          }
        ]
      },
      {
        "mealDate": "2026-09-07",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "마라샹궈★ 계란볶음밥"
          },
          {
            "cornerName": "특식",
            "displayOrder": 2,
            "menuName": "알리고치즈치킨스테이크 후리가께주먹밥"
          },
          {
            "cornerName": "특식",
            "displayOrder": 3,
            "menuName": "불고기비빔밥★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 4,
            "menuName": "순살돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 5,
            "menuName": "치즈돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 6,
            "menuName": "고구마돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 7,
            "menuName": "왕돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 8,
            "menuName": "왕치즈돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 9,
            "menuName": "라면"
          },
          {
            "cornerName": "특식",
            "displayOrder": 10,
            "menuName": "우동"
          },
          {
            "cornerName": "특식",
            "displayOrder": 11,
            "menuName": "우동밥"
          }
        ]
      },
      {
        "mealDate": "2026-09-07",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "1식4찬 자율배식"
          }
        ]
      }
    ]
  }
}
```

<details>
<summary>Unwrapped (`POST /api/v1/crawl/meals`) 형태 — data 본문만</summary>

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "정보센터식당",
  "sourceUrl": "https://coop.knu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07",
  "meals": [
    {
      "mealDate": "2026-09-05",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "특식"
        }
      ]
    },
    {
      "mealDate": "2026-09-05",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "특식"
        }
      ]
    },
    {
      "mealDate": "2026-09-06",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "특식"
        }
      ]
    },
    {
      "mealDate": "2026-09-06",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "특식"
        }
      ]
    },
    {
      "mealDate": "2026-09-07",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "마라샹궈★ 계란볶음밥"
        },
        {
          "cornerName": "특식",
          "displayOrder": 2,
          "menuName": "알리고치즈치킨스테이크 후리가께주먹밥"
        },
        {
          "cornerName": "특식",
          "displayOrder": 3,
          "menuName": "불고기비빔밥★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 4,
          "menuName": "순살돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 5,
          "menuName": "치즈돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 6,
          "menuName": "고구마돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 7,
          "menuName": "왕돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 8,
          "menuName": "왕치즈돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 9,
          "menuName": "라면"
        },
        {
          "cornerName": "특식",
          "displayOrder": 10,
          "menuName": "우동"
        },
        {
          "cornerName": "특식",
          "displayOrder": 11,
          "menuName": "우동밥"
        }
      ]
    },
    {
      "mealDate": "2026-09-07",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "1식4찬 자율배식"
        }
      ]
    }
  ]
}
```

</details>

</details>

<details>
<summary><strong>복지관 교직원식당</strong> — HTTP 200, meals=3 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "복지관 교직원식당",
  "sourceUrl": "https://coop.knu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "경북대학교",
    "cafeteriaName": "복지관 교직원식당",
    "sourceUrl": "https://coop.knu.ac.kr",
    "startDate": "2026-09-05",
    "endDate": "2026-09-07",
    "meals": [
      {
        "mealDate": "2026-09-05",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "정식"
          }
        ]
      },
      {
        "mealDate": "2026-09-06",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "정식"
          }
        ]
      },
      {
        "mealDate": "2026-09-07",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "콩나물밥*양념장 청국장찌개 중화풍돈육볶음★ 유자꿔바로우★ 참깨비빔우동 포기김치"
          }
        ]
      }
    ]
  }
}
```

<details>
<summary>Unwrapped (`POST /api/v1/crawl/meals`) 형태 — data 본문만</summary>

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "복지관 교직원식당",
  "sourceUrl": "https://coop.knu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07",
  "meals": [
    {
      "mealDate": "2026-09-05",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "정식"
        }
      ]
    },
    {
      "mealDate": "2026-09-06",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "정식"
        }
      ]
    },
    {
      "mealDate": "2026-09-07",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "콩나물밥*양념장 청국장찌개 중화풍돈육볶음★ 유자꿔바로우★ 참깨비빔우동 포기김치"
        }
      ]
    }
  ]
}
```

</details>

</details>

<details>
<summary><strong>카페테리아 첨성</strong> — HTTP 200, meals=6 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "카페테리아 첨성",
  "sourceUrl": "https://coop.knu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "경북대학교",
    "cafeteriaName": "카페테리아 첨성",
    "sourceUrl": "https://coop.knu.ac.kr",
    "startDate": "2026-09-05",
    "endDate": "2026-09-07",
    "meals": [
      {
        "mealDate": "2026-09-05",
        "mealType": "BREAKFAST",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "특식"
          }
        ]
      },
      {
        "mealDate": "2026-09-05",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "특식"
          }
        ]
      },
      {
        "mealDate": "2026-09-06",
        "mealType": "BREAKFAST",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "특식"
          }
        ]
      },
      {
        "mealDate": "2026-09-06",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "특식"
          }
        ]
      },
      {
        "mealDate": "2026-09-07",
        "mealType": "BREAKFAST",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "천원의 아침밥"
          }
        ]
      },
      {
        "mealDate": "2026-09-07",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "필라프& 불고기베이크 &스프"
          },
          {
            "cornerName": "특식",
            "displayOrder": 2,
            "menuName": "참치비빔밥"
          },
          {
            "cornerName": "특식",
            "displayOrder": 3,
            "menuName": "순살돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 4,
            "menuName": "치즈돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 5,
            "menuName": "고구마돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 6,
            "menuName": "왕돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 7,
            "menuName": "치즈왕돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 8,
            "menuName": "라면"
          }
        ]
      }
    ]
  }
}
```

<details>
<summary>Unwrapped (`POST /api/v1/crawl/meals`) 형태 — data 본문만</summary>

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "카페테리아 첨성",
  "sourceUrl": "https://coop.knu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07",
  "meals": [
    {
      "mealDate": "2026-09-05",
      "mealType": "BREAKFAST",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "특식"
        }
      ]
    },
    {
      "mealDate": "2026-09-05",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "특식"
        }
      ]
    },
    {
      "mealDate": "2026-09-06",
      "mealType": "BREAKFAST",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "특식"
        }
      ]
    },
    {
      "mealDate": "2026-09-06",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "특식"
        }
      ]
    },
    {
      "mealDate": "2026-09-07",
      "mealType": "BREAKFAST",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "천원의 아침밥"
        }
      ]
    },
    {
      "mealDate": "2026-09-07",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "필라프& 불고기베이크 &스프"
        },
        {
          "cornerName": "특식",
          "displayOrder": 2,
          "menuName": "참치비빔밥"
        },
        {
          "cornerName": "특식",
          "displayOrder": 3,
          "menuName": "순살돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 4,
          "menuName": "치즈돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 5,
          "menuName": "고구마돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 6,
          "menuName": "왕돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 7,
          "menuName": "치즈왕돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 8,
          "menuName": "라면"
        }
      ]
    }
  ]
}
```

</details>

</details>

<details>
<summary><strong>GP감꽃식당</strong> — HTTP 200, meals=3 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "GP감꽃식당",
  "sourceUrl": "https://coop.knu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "경북대학교",
    "cafeteriaName": "GP감꽃식당",
    "sourceUrl": "https://coop.knu.ac.kr",
    "startDate": "2026-09-05",
    "endDate": "2026-09-07",
    "meals": [
      {
        "mealDate": "2026-09-05",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "정식"
          }
        ]
      },
      {
        "mealDate": "2026-09-06",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "정식"
          }
        ]
      },
      {
        "mealDate": "2026-09-07",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "백미밥 닭개장 마파두부★ 고구마무스콘치즈펜네 망고샐러드 포기김치"
          }
        ]
      }
    ]
  }
}
```

<details>
<summary>Unwrapped (`POST /api/v1/crawl/meals`) 형태 — data 본문만</summary>

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "GP감꽃식당",
  "sourceUrl": "https://coop.knu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07",
  "meals": [
    {
      "mealDate": "2026-09-05",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "정식"
        }
      ]
    },
    {
      "mealDate": "2026-09-06",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "정식"
        }
      ]
    },
    {
      "mealDate": "2026-09-07",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "백미밥 닭개장 마파두부★ 고구마무스콘치즈펜네 망고샐러드 포기김치"
        }
      ]
    }
  ]
}
```

</details>

</details>

<details>
<summary><strong>공학관교직원식당(외부업체)</strong> — HTTP 200, meals=6 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "공학관교직원식당(외부업체)",
  "sourceUrl": "https://coop.knu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "경북대학교",
    "cafeteriaName": "공학관교직원식당(외부업체)",
    "sourceUrl": "https://coop.knu.ac.kr",
    "startDate": "2026-09-05",
    "endDate": "2026-09-07",
    "meals": [
      {
        "mealDate": "2026-09-05",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "정식"
          }
        ]
      },
      {
        "mealDate": "2026-09-05",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "정식"
          }
        ]
      },
      {
        "mealDate": "2026-09-06",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "정식"
          }
        ]
      },
      {
        "mealDate": "2026-09-06",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "정식"
          }
        ]
      },
      {
        "mealDate": "2026-09-07",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "흑미밥/쌀밥 육개장 닭볶음탕 감자고르케&케찹 멸치마늘쫑볶음 직접담근김치 잔치국수"
          }
        ]
      },
      {
        "mealDate": "2026-09-07",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "기장밥/쌀밥 조개살부추국 돈가스카레★ 야채쫄면 우엉채땅콩조림 직접담근김치 도시락김"
          }
        ]
      }
    ]
  }
}
```

<details>
<summary>Unwrapped (`POST /api/v1/crawl/meals`) 형태 — data 본문만</summary>

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "공학관교직원식당(외부업체)",
  "sourceUrl": "https://coop.knu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07",
  "meals": [
    {
      "mealDate": "2026-09-05",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "정식"
        }
      ]
    },
    {
      "mealDate": "2026-09-05",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "정식"
        }
      ]
    },
    {
      "mealDate": "2026-09-06",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "정식"
        }
      ]
    },
    {
      "mealDate": "2026-09-06",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "정식"
        }
      ]
    },
    {
      "mealDate": "2026-09-07",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "흑미밥/쌀밥 육개장 닭볶음탕 감자고르케&케찹 멸치마늘쫑볶음 직접담근김치 잔치국수"
        }
      ]
    },
    {
      "mealDate": "2026-09-07",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "기장밥/쌀밥 조개살부추국 돈가스카레★ 야채쫄면 우엉채땅콩조림 직접담근김치 도시락김"
        }
      ]
    }
  ]
}
```

</details>

</details>

---

## 국립경국대학교

<details>
<summary><strong>이룸관(안동, 학생식당)</strong> — HTTP 200, meals=2 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "국립경국대학교",
  "cafeteriaName": "이룸관(안동, 학생식당)",
  "sourceUrl": "https://www.gknu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "국립경국대학교",
    "cafeteriaName": "이룸관(안동, 학생식당)",
    "sourceUrl": "https://www.gknu.ac.kr",
    "startDate": "2026-09-05",
    "endDate": "2026-09-07",
    "meals": [
      {
        "mealDate": "2026-09-07",
        "mealType": "BREAKFAST",
        "menus": [
          {
            "cornerName": "조식",
            "displayOrder": 1,
            "menuName": "[천원의 아침밥 (채움관에서 운영)]"
          },
          {
            "cornerName": "조식",
            "displayOrder": 2,
            "menuName": "찰흑미밥"
          },
          {
            "cornerName": "조식",
            "displayOrder": 3,
            "menuName": "계란국"
          },
          {
            "cornerName": "조식",
            "displayOrder": 4,
            "menuName": "꿔바로우"
          },
          {
            "cornerName": "조식",
            "displayOrder": 5,
            "menuName": "동태두부조림"
          },
          {
            "cornerName": "조식",
            "displayOrder": 6,
            "menuName": "오이무침"
          },
          {
            "cornerName": "조식",
            "displayOrder": 7,
            "menuName": "배추김치/ 마시는요거트"
          },
          {
            "cornerName": "조식",
            "displayOrder": 8,
            "menuName": "or"
          },
          {
            "cornerName": "조식",
            "displayOrder": 9,
            "menuName": "왕김밥"
          },
          {
            "cornerName": "조식",
            "displayOrder": 10,
            "menuName": "마시는요거트"
          },
          {
            "cornerName": "조식",
            "displayOrder": 11,
            "menuName": "바나나"
          }
        ]
      },
      {
        "mealDate": "2026-09-07",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "중식",
            "displayOrder": 1,
            "menuName": "찰흑미밥"
          },
          {
            "cornerName": "중식",
            "displayOrder": 2,
            "menuName": "짬뽕순두부찌개"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "파인애플닭살겨자무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "쫄면야채무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "볼어묵감자조림"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "열무김치"
          },
          {
            "cornerName": "중식",
            "displayOrder": 7,
            "menuName": "배추김치/요구르트"
          }
        ]
      }
    ]
  }
}
```

<details>
<summary>Unwrapped (`POST /api/v1/crawl/meals`) 형태 — data 본문만</summary>

```json
{
  "schoolName": "국립경국대학교",
  "cafeteriaName": "이룸관(안동, 학생식당)",
  "sourceUrl": "https://www.gknu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07",
  "meals": [
    {
      "mealDate": "2026-09-07",
      "mealType": "BREAKFAST",
      "menus": [
        {
          "cornerName": "조식",
          "displayOrder": 1,
          "menuName": "[천원의 아침밥 (채움관에서 운영)]"
        },
        {
          "cornerName": "조식",
          "displayOrder": 2,
          "menuName": "찰흑미밥"
        },
        {
          "cornerName": "조식",
          "displayOrder": 3,
          "menuName": "계란국"
        },
        {
          "cornerName": "조식",
          "displayOrder": 4,
          "menuName": "꿔바로우"
        },
        {
          "cornerName": "조식",
          "displayOrder": 5,
          "menuName": "동태두부조림"
        },
        {
          "cornerName": "조식",
          "displayOrder": 6,
          "menuName": "오이무침"
        },
        {
          "cornerName": "조식",
          "displayOrder": 7,
          "menuName": "배추김치/ 마시는요거트"
        },
        {
          "cornerName": "조식",
          "displayOrder": 8,
          "menuName": "or"
        },
        {
          "cornerName": "조식",
          "displayOrder": 9,
          "menuName": "왕김밥"
        },
        {
          "cornerName": "조식",
          "displayOrder": 10,
          "menuName": "마시는요거트"
        },
        {
          "cornerName": "조식",
          "displayOrder": 11,
          "menuName": "바나나"
        }
      ]
    },
    {
      "mealDate": "2026-09-07",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "중식",
          "displayOrder": 1,
          "menuName": "찰흑미밥"
        },
        {
          "cornerName": "중식",
          "displayOrder": 2,
          "menuName": "짬뽕순두부찌개"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "파인애플닭살겨자무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "쫄면야채무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "볼어묵감자조림"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "열무김치"
        },
        {
          "cornerName": "중식",
          "displayOrder": 7,
          "menuName": "배추김치/요구르트"
        }
      ]
    }
  ]
}
```

</details>

</details>

<details>
<summary><strong>채움관(안동, 교직원식당)</strong> — HTTP 200, meals=2 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "국립경국대학교",
  "cafeteriaName": "채움관(안동, 교직원식당)",
  "sourceUrl": "https://www.gknu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "국립경국대학교",
    "cafeteriaName": "채움관(안동, 교직원식당)",
    "sourceUrl": "https://www.gknu.ac.kr",
    "startDate": "2026-09-05",
    "endDate": "2026-09-07",
    "meals": [
      {
        "mealDate": "2026-09-07",
        "mealType": "BREAKFAST",
        "menus": [
          {
            "cornerName": "조식",
            "displayOrder": 1,
            "menuName": "[천원의 아침밥]"
          },
          {
            "cornerName": "조식",
            "displayOrder": 2,
            "menuName": "찰흑미밥"
          },
          {
            "cornerName": "조식",
            "displayOrder": 3,
            "menuName": "계란국"
          },
          {
            "cornerName": "조식",
            "displayOrder": 4,
            "menuName": "꿔바로우"
          },
          {
            "cornerName": "조식",
            "displayOrder": 5,
            "menuName": "동태두부조림"
          },
          {
            "cornerName": "조식",
            "displayOrder": 6,
            "menuName": "오이무침"
          },
          {
            "cornerName": "조식",
            "displayOrder": 7,
            "menuName": "배추김치/ 마시는요거트"
          },
          {
            "cornerName": "조식",
            "displayOrder": 8,
            "menuName": "or"
          },
          {
            "cornerName": "조식",
            "displayOrder": 9,
            "menuName": "왕김밥"
          },
          {
            "cornerName": "조식",
            "displayOrder": 10,
            "menuName": "마시는요거트"
          },
          {
            "cornerName": "조식",
            "displayOrder": 11,
            "menuName": "바나나"
          }
        ]
      },
      {
        "mealDate": "2026-09-07",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "중식",
            "displayOrder": 1,
            "menuName": "찰흑미밥"
          },
          {
            "cornerName": "중식",
            "displayOrder": 2,
            "menuName": "짬뽕순두부찌개"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "파인애플닭살겨자무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "쫄면야채무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "볼어묵감자조림"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "열무김치"
          },
          {
            "cornerName": "중식",
            "displayOrder": 7,
            "menuName": "배추김치/요구르트"
          }
        ]
      }
    ]
  }
}
```

<details>
<summary>Unwrapped (`POST /api/v1/crawl/meals`) 형태 — data 본문만</summary>

```json
{
  "schoolName": "국립경국대학교",
  "cafeteriaName": "채움관(안동, 교직원식당)",
  "sourceUrl": "https://www.gknu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07",
  "meals": [
    {
      "mealDate": "2026-09-07",
      "mealType": "BREAKFAST",
      "menus": [
        {
          "cornerName": "조식",
          "displayOrder": 1,
          "menuName": "[천원의 아침밥]"
        },
        {
          "cornerName": "조식",
          "displayOrder": 2,
          "menuName": "찰흑미밥"
        },
        {
          "cornerName": "조식",
          "displayOrder": 3,
          "menuName": "계란국"
        },
        {
          "cornerName": "조식",
          "displayOrder": 4,
          "menuName": "꿔바로우"
        },
        {
          "cornerName": "조식",
          "displayOrder": 5,
          "menuName": "동태두부조림"
        },
        {
          "cornerName": "조식",
          "displayOrder": 6,
          "menuName": "오이무침"
        },
        {
          "cornerName": "조식",
          "displayOrder": 7,
          "menuName": "배추김치/ 마시는요거트"
        },
        {
          "cornerName": "조식",
          "displayOrder": 8,
          "menuName": "or"
        },
        {
          "cornerName": "조식",
          "displayOrder": 9,
          "menuName": "왕김밥"
        },
        {
          "cornerName": "조식",
          "displayOrder": 10,
          "menuName": "마시는요거트"
        },
        {
          "cornerName": "조식",
          "displayOrder": 11,
          "menuName": "바나나"
        }
      ]
    },
    {
      "mealDate": "2026-09-07",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "중식",
          "displayOrder": 1,
          "menuName": "찰흑미밥"
        },
        {
          "cornerName": "중식",
          "displayOrder": 2,
          "menuName": "짬뽕순두부찌개"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "파인애플닭살겨자무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "쫄면야채무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "볼어묵감자조림"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "열무김치"
        },
        {
          "cornerName": "중식",
          "displayOrder": 7,
          "menuName": "배추김치/요구르트"
        }
      ]
    }
  ]
}
```

</details>

</details>

<details>
<summary><strong>양식코너(안동)</strong> — HTTP 200, meals=1 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "국립경국대학교",
  "cafeteriaName": "양식코너(안동)",
  "sourceUrl": "https://www.gknu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "국립경국대학교",
    "cafeteriaName": "양식코너(안동)",
    "sourceUrl": "https://www.gknu.ac.kr",
    "startDate": "2026-09-05",
    "endDate": "2026-09-07",
    "meals": [
      {
        "mealDate": "2026-09-07",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "양식코너",
            "displayOrder": 1,
            "menuName": "제주흑돼지김치찌개"
          },
          {
            "cornerName": "양식코너",
            "displayOrder": 2,
            "menuName": "참치김치찌개"
          },
          {
            "cornerName": "양식코너",
            "displayOrder": 3,
            "menuName": "스팸김치찌개"
          },
          {
            "cornerName": "양식코너",
            "displayOrder": 4,
            "menuName": "돈가스마요덮밥"
          },
          {
            "cornerName": "양식코너",
            "displayOrder": 5,
            "menuName": "육회비빔밥"
          },
          {
            "cornerName": "양식코너",
            "displayOrder": 6,
            "menuName": "통스팸김치덮밥"
          },
          {
            "cornerName": "양식코너",
            "displayOrder": 7,
            "menuName": "삼겹덮밥"
          },
          {
            "cornerName": "양식코너",
            "displayOrder": 8,
            "menuName": "간장불고기덮밥"
          },
          {
            "cornerName": "양식코너",
            "displayOrder": 9,
            "menuName": "고추장불고기덮밥"
          },
          {
            "cornerName": "양식코너",
            "displayOrder": 10,
            "menuName": "명란에그덮밥"
          },
          {
            "cornerName": "양식코너",
            "displayOrder": 11,
            "menuName": "스팸에그마요덮밥"
          },
          {
            "cornerName": "양식코너",
            "displayOrder": 12,
            "menuName": "핵불닭덮밥"
          },
          {
            "cornerName": "양식코너",
            "displayOrder": 13,
            "menuName": "가라아게에그덮밥"
          },
          {
            "cornerName": "양식코너",
            "displayOrder": 14,
            "menuName": "우삼겹비빔밥"
          },
          {
            "cornerName": "양식코너",
            "displayOrder": 15,
            "menuName": "참치비빔밥"
          }
        ]
      }
    ]
  }
}
```

<details>
<summary>Unwrapped (`POST /api/v1/crawl/meals`) 형태 — data 본문만</summary>

```json
{
  "schoolName": "국립경국대학교",
  "cafeteriaName": "양식코너(안동)",
  "sourceUrl": "https://www.gknu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07",
  "meals": [
    {
      "mealDate": "2026-09-07",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "양식코너",
          "displayOrder": 1,
          "menuName": "제주흑돼지김치찌개"
        },
        {
          "cornerName": "양식코너",
          "displayOrder": 2,
          "menuName": "참치김치찌개"
        },
        {
          "cornerName": "양식코너",
          "displayOrder": 3,
          "menuName": "스팸김치찌개"
        },
        {
          "cornerName": "양식코너",
          "displayOrder": 4,
          "menuName": "돈가스마요덮밥"
        },
        {
          "cornerName": "양식코너",
          "displayOrder": 5,
          "menuName": "육회비빔밥"
        },
        {
          "cornerName": "양식코너",
          "displayOrder": 6,
          "menuName": "통스팸김치덮밥"
        },
        {
          "cornerName": "양식코너",
          "displayOrder": 7,
          "menuName": "삼겹덮밥"
        },
        {
          "cornerName": "양식코너",
          "displayOrder": 8,
          "menuName": "간장불고기덮밥"
        },
        {
          "cornerName": "양식코너",
          "displayOrder": 9,
          "menuName": "고추장불고기덮밥"
        },
        {
          "cornerName": "양식코너",
          "displayOrder": 10,
          "menuName": "명란에그덮밥"
        },
        {
          "cornerName": "양식코너",
          "displayOrder": 11,
          "menuName": "스팸에그마요덮밥"
        },
        {
          "cornerName": "양식코너",
          "displayOrder": 12,
          "menuName": "핵불닭덮밥"
        },
        {
          "cornerName": "양식코너",
          "displayOrder": 13,
          "menuName": "가라아게에그덮밥"
        },
        {
          "cornerName": "양식코너",
          "displayOrder": 14,
          "menuName": "우삼겹비빔밥"
        },
        {
          "cornerName": "양식코너",
          "displayOrder": 15,
          "menuName": "참치비빔밥"
        }
      ]
    }
  ]
}
```

</details>

</details>

<details>
<summary><strong>학생식당(예천)</strong> — HTTP 200, meals=7 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "국립경국대학교",
  "cafeteriaName": "학생식당(예천)",
  "sourceUrl": "https://www.gknu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "국립경국대학교",
    "cafeteriaName": "학생식당(예천)",
    "sourceUrl": "https://www.gknu.ac.kr",
    "startDate": "2026-09-05",
    "endDate": "2026-09-07",
    "meals": [
      {
        "mealDate": "2026-09-05",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "중식",
            "displayOrder": 1,
            "menuName": "열무비빔국수"
          },
          {
            "cornerName": "중식",
            "displayOrder": 2,
            "menuName": "햄주먹밥"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "뿌링핫도그"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "명엽채볶음"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "단무지"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "바나나우유"
          }
        ]
      },
      {
        "mealDate": "2026-09-05",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "석식",
            "displayOrder": 1,
            "menuName": "쌀밥"
          },
          {
            "cornerName": "석식",
            "displayOrder": 2,
            "menuName": "두부참치찌개"
          },
          {
            "cornerName": "석식",
            "displayOrder": 3,
            "menuName": "닭살데리야끼조림"
          },
          {
            "cornerName": "석식",
            "displayOrder": 4,
            "menuName": "만두찜*양념장"
          },
          {
            "cornerName": "석식",
            "displayOrder": 5,
            "menuName": "미역초무침"
          },
          {
            "cornerName": "석식",
            "displayOrder": 6,
            "menuName": "무말랭이무침"
          }
        ]
      },
      {
        "mealDate": "2026-09-06",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "중식",
            "displayOrder": 1,
            "menuName": "쌀밥"
          },
          {
            "cornerName": "중식",
            "displayOrder": 2,
            "menuName": "순살감자탕"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "오징어까스*머스타드"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "멸치볶음"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "오이쌈장무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "배추김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-06",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "석식",
            "displayOrder": 1,
            "menuName": "하이라이스"
          },
          {
            "cornerName": "석식",
            "displayOrder": 2,
            "menuName": "계란국"
          },
          {
            "cornerName": "석식",
            "displayOrder": 3,
            "menuName": "깐풍기"
          },
          {
            "cornerName": "석식",
            "displayOrder": 4,
            "menuName": "비엔나채소볶음"
          },
          {
            "cornerName": "석식",
            "displayOrder": 5,
            "menuName": "깍두기"
          },
          {
            "cornerName": "석식",
            "displayOrder": 6,
            "menuName": "모둠견과"
          }
        ]
      },
      {
        "mealDate": "2026-09-07",
        "mealType": "BREAKFAST",
        "menus": [
          {
            "cornerName": "조식",
            "displayOrder": 1,
            "menuName": "[천원의 아침밥]"
          },
          {
            "cornerName": "조식",
            "displayOrder": 2,
            "menuName": "쌀밥/쇠고기국밥"
          },
          {
            "cornerName": "조식",
            "displayOrder": 3,
            "menuName": "주꾸미볶음"
          },
          {
            "cornerName": "조식",
            "displayOrder": 4,
            "menuName": "계란장조림"
          },
          {
            "cornerName": "조식",
            "displayOrder": 5,
            "menuName": "오이생채"
          },
          {
            "cornerName": "조식",
            "displayOrder": 6,
            "menuName": "배추김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-07",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "중식",
            "displayOrder": 1,
            "menuName": "잡곡밥"
          },
          {
            "cornerName": "중식",
            "displayOrder": 2,
            "menuName": "육개장"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "훈제오리볶음"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "두부찜&볶음김치"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "상추겉절이"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "총각김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-07",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "석식",
            "displayOrder": 1,
            "menuName": "미트소스스파게티"
          },
          {
            "cornerName": "석식",
            "displayOrder": 2,
            "menuName": "치킨가스&소스"
          },
          {
            "cornerName": "석식",
            "displayOrder": 3,
            "menuName": "열대과일샐러드&키위D"
          },
          {
            "cornerName": "석식",
            "displayOrder": 4,
            "menuName": "오이피클"
          },
          {
            "cornerName": "석식",
            "displayOrder": 5,
            "menuName": "미니탄산"
          },
          {
            "cornerName": "석식",
            "displayOrder": 6,
            "menuName": "카스테라빵"
          }
        ]
      }
    ]
  }
}
```

<details>
<summary>Unwrapped (`POST /api/v1/crawl/meals`) 형태 — data 본문만</summary>

```json
{
  "schoolName": "국립경국대학교",
  "cafeteriaName": "학생식당(예천)",
  "sourceUrl": "https://www.gknu.ac.kr",
  "startDate": "2026-09-05",
  "endDate": "2026-09-07",
  "meals": [
    {
      "mealDate": "2026-09-05",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "중식",
          "displayOrder": 1,
          "menuName": "열무비빔국수"
        },
        {
          "cornerName": "중식",
          "displayOrder": 2,
          "menuName": "햄주먹밥"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "뿌링핫도그"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "명엽채볶음"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "단무지"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "바나나우유"
        }
      ]
    },
    {
      "mealDate": "2026-09-05",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "석식",
          "displayOrder": 1,
          "menuName": "쌀밥"
        },
        {
          "cornerName": "석식",
          "displayOrder": 2,
          "menuName": "두부참치찌개"
        },
        {
          "cornerName": "석식",
          "displayOrder": 3,
          "menuName": "닭살데리야끼조림"
        },
        {
          "cornerName": "석식",
          "displayOrder": 4,
          "menuName": "만두찜*양념장"
        },
        {
          "cornerName": "석식",
          "displayOrder": 5,
          "menuName": "미역초무침"
        },
        {
          "cornerName": "석식",
          "displayOrder": 6,
          "menuName": "무말랭이무침"
        }
      ]
    },
    {
      "mealDate": "2026-09-06",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "중식",
          "displayOrder": 1,
          "menuName": "쌀밥"
        },
        {
          "cornerName": "중식",
          "displayOrder": 2,
          "menuName": "순살감자탕"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "오징어까스*머스타드"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "멸치볶음"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "오이쌈장무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "배추김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-06",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "석식",
          "displayOrder": 1,
          "menuName": "하이라이스"
        },
        {
          "cornerName": "석식",
          "displayOrder": 2,
          "menuName": "계란국"
        },
        {
          "cornerName": "석식",
          "displayOrder": 3,
          "menuName": "깐풍기"
        },
        {
          "cornerName": "석식",
          "displayOrder": 4,
          "menuName": "비엔나채소볶음"
        },
        {
          "cornerName": "석식",
          "displayOrder": 5,
          "menuName": "깍두기"
        },
        {
          "cornerName": "석식",
          "displayOrder": 6,
          "menuName": "모둠견과"
        }
      ]
    },
    {
      "mealDate": "2026-09-07",
      "mealType": "BREAKFAST",
      "menus": [
        {
          "cornerName": "조식",
          "displayOrder": 1,
          "menuName": "[천원의 아침밥]"
        },
        {
          "cornerName": "조식",
          "displayOrder": 2,
          "menuName": "쌀밥/쇠고기국밥"
        },
        {
          "cornerName": "조식",
          "displayOrder": 3,
          "menuName": "주꾸미볶음"
        },
        {
          "cornerName": "조식",
          "displayOrder": 4,
          "menuName": "계란장조림"
        },
        {
          "cornerName": "조식",
          "displayOrder": 5,
          "menuName": "오이생채"
        },
        {
          "cornerName": "조식",
          "displayOrder": 6,
          "menuName": "배추김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-07",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "중식",
          "displayOrder": 1,
          "menuName": "잡곡밥"
        },
        {
          "cornerName": "중식",
          "displayOrder": 2,
          "menuName": "육개장"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "훈제오리볶음"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "두부찜&볶음김치"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "상추겉절이"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "총각김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-07",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "석식",
          "displayOrder": 1,
          "menuName": "미트소스스파게티"
        },
        {
          "cornerName": "석식",
          "displayOrder": 2,
          "menuName": "치킨가스&소스"
        },
        {
          "cornerName": "석식",
          "displayOrder": 3,
          "menuName": "열대과일샐러드&키위D"
        },
        {
          "cornerName": "석식",
          "displayOrder": 4,
          "menuName": "오이피클"
        },
        {
          "cornerName": "석식",
          "displayOrder": 5,
          "menuName": "미니탄산"
        },
        {
          "cornerName": "석식",
          "displayOrder": 6,
          "menuName": "카스테라빵"
        }
      ]
    }
  ]
}
```

</details>

</details>

---

