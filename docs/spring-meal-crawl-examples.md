# Spring 식단 크롤 API — 실제 요청/응답 예시

Spring 연동 확인용으로, **실제로 호출한 요청·응답**을 기록한 문서입니다.

## 공통

| 항목 | 값 |
|---|---|
| Wrapped 엔드포인트 | `POST /api/v1/python/meals/crawl` |
| Unwrapped (Spring) | `POST /api/v1/crawl/meals` — 아래 응답의 `data` 본문만 반환 |
| 조회 기간 | `2026-08-31` ~ `2026-09-04` (해당 주 월~금) |
| Accept-Language | `ko` |

> 경북대·경국대 `sourceUrl`은 **짧은 형태**(쿼리만)를 사용했습니다. 서버가 정규 경로로 보정합니다.

## 요약

| 학교 | 식당 | HTTP | meals 수 |
|---|---|---|---|
| 금오공과대학교 | 일품식당 | 200 | 15 |
| 금오공과대학교 | 정찬식당 | 200 | 5 |
| 금오공과대학교 | 분식당 | 200 | 10 |
| 경북대학교 | 정보센터식당 | 200 | 10 |
| 경북대학교 | 복지관 교직원식당 | 200 | 5 |
| 경북대학교 | 카페테리아 첨성 | 200 | 10 |
| 경북대학교 | GP감꽃식당 | 200 | 5 |
| 경북대학교 | 공학관교직원식당(외부업체) | 200 | 10 |
| 국립경국대학교 | 이룸관(안동, 학생식당) | 200 | 9 |
| 국립경국대학교 | 채움관(안동, 교직원식당) | 200 | 9 |
| 국립경국대학교 | 양식코너(안동) | 200 | 5 |
| 국립경국대학교 | 학생식당(예천) | 200 | 14 |

---

## 금오공과대학교

<details>
<summary><strong>일품식당</strong> — HTTP 200, meals=15 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "금오공과대학교",
  "cafeteriaName": "일품식당",
  "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant01.do",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04"
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
    "startDate": "2026-08-31",
    "endDate": "2026-09-04",
    "meals": [
      {
        "mealDate": "2026-08-31",
        "mealType": "BREAKFAST",
        "menus": [
          {
            "cornerName": "조식",
            "displayOrder": 1,
            "menuName": "-미운영-"
          }
        ]
      },
      {
        "mealDate": "2026-08-31",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "라면류"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 2,
            "menuName": "돈가스류"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 3,
            "menuName": "육회비빔밥"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 4,
            "menuName": "일품정식(석식)"
          }
        ]
      },
      {
        "mealDate": "2026-08-31",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "라면류"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 2,
            "menuName": "돈가스류"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 3,
            "menuName": "육회비빔밥"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 4,
            "menuName": "일품정식(석식)"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
        "mealType": "BREAKFAST",
        "menus": [
          {
            "cornerName": "조식",
            "displayOrder": 1,
            "menuName": "든든한제육&돈까스정식"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "일품정식(석식)"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "일품정식(석식)"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
        "mealType": "BREAKFAST",
        "menus": [
          {
            "cornerName": "조식",
            "displayOrder": 1,
            "menuName": "맛살튀김줄김밥"
          },
          {
            "cornerName": "조식",
            "displayOrder": 2,
            "menuName": "더블햄토마토샌드위치"
          },
          {
            "cornerName": "조식",
            "displayOrder": 3,
            "menuName": "홈요거트"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "일품정식(석식)"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "일품정식(석식)"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
        "mealType": "BREAKFAST",
        "menus": [
          {
            "cornerName": "조식",
            "displayOrder": 1,
            "menuName": "불고기반숙비빔밥"
          },
          {
            "cornerName": "조식",
            "displayOrder": 2,
            "menuName": "딸기요플레"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "등촌샤브칼국수"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 2,
            "menuName": "일품정식(석식)"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "등촌샤브칼국수"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 2,
            "menuName": "일품정식(석식)"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
        "mealType": "BREAKFAST",
        "menus": [
          {
            "cornerName": "조식",
            "displayOrder": 1,
            "menuName": "햄쌈반반제육정찬"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "치킨크림스튜우동"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 2,
            "menuName": "일품정식(석식)"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "치킨크림스튜우동"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 2,
            "menuName": "일품정식(석식)"
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
  "startDate": "2026-08-31",
  "endDate": "2026-09-04",
  "meals": [
    {
      "mealDate": "2026-08-31",
      "mealType": "BREAKFAST",
      "menus": [
        {
          "cornerName": "조식",
          "displayOrder": 1,
          "menuName": "-미운영-"
        }
      ]
    },
    {
      "mealDate": "2026-08-31",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "라면류"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 2,
          "menuName": "돈가스류"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 3,
          "menuName": "육회비빔밥"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 4,
          "menuName": "일품정식(석식)"
        }
      ]
    },
    {
      "mealDate": "2026-08-31",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "라면류"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 2,
          "menuName": "돈가스류"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 3,
          "menuName": "육회비빔밥"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 4,
          "menuName": "일품정식(석식)"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
      "mealType": "BREAKFAST",
      "menus": [
        {
          "cornerName": "조식",
          "displayOrder": 1,
          "menuName": "든든한제육&돈까스정식"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "일품정식(석식)"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "일품정식(석식)"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
      "mealType": "BREAKFAST",
      "menus": [
        {
          "cornerName": "조식",
          "displayOrder": 1,
          "menuName": "맛살튀김줄김밥"
        },
        {
          "cornerName": "조식",
          "displayOrder": 2,
          "menuName": "더블햄토마토샌드위치"
        },
        {
          "cornerName": "조식",
          "displayOrder": 3,
          "menuName": "홈요거트"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "일품정식(석식)"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "일품정식(석식)"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
      "mealType": "BREAKFAST",
      "menus": [
        {
          "cornerName": "조식",
          "displayOrder": 1,
          "menuName": "불고기반숙비빔밥"
        },
        {
          "cornerName": "조식",
          "displayOrder": 2,
          "menuName": "딸기요플레"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "등촌샤브칼국수"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 2,
          "menuName": "일품정식(석식)"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "등촌샤브칼국수"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 2,
          "menuName": "일품정식(석식)"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
      "mealType": "BREAKFAST",
      "menus": [
        {
          "cornerName": "조식",
          "displayOrder": 1,
          "menuName": "햄쌈반반제육정찬"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "치킨크림스튜우동"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 2,
          "menuName": "일품정식(석식)"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "치킨크림스튜우동"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 2,
          "menuName": "일품정식(석식)"
        }
      ]
    }
  ]
}
```

</details>

</details>

<details>
<summary><strong>정찬식당</strong> — HTTP 200, meals=5 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "금오공과대학교",
  "cafeteriaName": "정찬식당",
  "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant02.do",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04"
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
    "startDate": "2026-08-31",
    "endDate": "2026-09-04",
    "meals": [
      {
        "mealDate": "2026-08-31",
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
            "menuName": "꽃게된장국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "훈제오리야채찜"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "매콤버섯볶음"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "병아리콩우엉조림"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "블루베리샐러드"
          },
          {
            "cornerName": "중식",
            "displayOrder": 7,
            "menuName": "배추김치"
          },
          {
            "cornerName": "중식",
            "displayOrder": 8,
            "menuName": "------------------"
          },
          {
            "cornerName": "중식",
            "displayOrder": 9,
            "menuName": "(11:00~11:40)"
          },
          {
            "cornerName": "중식",
            "displayOrder": 10,
            "menuName": "일품식당에서"
          },
          {
            "cornerName": "중식",
            "displayOrder": 11,
            "menuName": "위의 정식메뉴 운영"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "중식",
            "displayOrder": 1,
            "menuName": "기장밥"
          },
          {
            "cornerName": "중식",
            "displayOrder": 2,
            "menuName": "전복갈비탕"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "충무식오징어어묵무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "청포묵김가루무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "고추소박이"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "열대과일샐러드"
          },
          {
            "cornerName": "중식",
            "displayOrder": 7,
            "menuName": "깍두기"
          },
          {
            "cornerName": "중식",
            "displayOrder": 8,
            "menuName": "------------------"
          },
          {
            "cornerName": "중식",
            "displayOrder": 9,
            "menuName": "(11:00~11:40)"
          },
          {
            "cornerName": "중식",
            "displayOrder": 10,
            "menuName": "일품식당에서"
          },
          {
            "cornerName": "중식",
            "displayOrder": 11,
            "menuName": "위의 정식메뉴 운영"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
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
            "menuName": "북어해장국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "닭갈비순대볶음"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "이연복가지찜"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "콩나물파채무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "삼색쌈무"
          },
          {
            "cornerName": "중식",
            "displayOrder": 7,
            "menuName": "배추김치"
          },
          {
            "cornerName": "중식",
            "displayOrder": 8,
            "menuName": "------------------"
          },
          {
            "cornerName": "중식",
            "displayOrder": 9,
            "menuName": "(11:00~11:40)"
          },
          {
            "cornerName": "중식",
            "displayOrder": 10,
            "menuName": "일품식당에서"
          },
          {
            "cornerName": "중식",
            "displayOrder": 11,
            "menuName": "위의 정식메뉴 운영"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
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
            "menuName": "열무된장국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "어린잎순두부"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "고구마고로케"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "그래놀라샐러드"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "깍두기"
          },
          {
            "cornerName": "중식",
            "displayOrder": 7,
            "menuName": "------------------"
          },
          {
            "cornerName": "중식",
            "displayOrder": 8,
            "menuName": "(11:00~11:40)"
          },
          {
            "cornerName": "중식",
            "displayOrder": 9,
            "menuName": "일품식당에서"
          },
          {
            "cornerName": "중식",
            "displayOrder": 10,
            "menuName": "위의 정식메뉴 운영"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
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
            "menuName": "조랭이미역국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "옥수수계란찜"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "마늘쫑건새우볶음"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "빵가루마요샐러드"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "배추김치"
          },
          {
            "cornerName": "중식",
            "displayOrder": 7,
            "menuName": "------------------"
          },
          {
            "cornerName": "중식",
            "displayOrder": 8,
            "menuName": "(11:00~11:40)"
          },
          {
            "cornerName": "중식",
            "displayOrder": 9,
            "menuName": "일품식당에서"
          },
          {
            "cornerName": "중식",
            "displayOrder": 10,
            "menuName": "위의 정식메뉴 운영"
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
  "startDate": "2026-08-31",
  "endDate": "2026-09-04",
  "meals": [
    {
      "mealDate": "2026-08-31",
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
          "menuName": "꽃게된장국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "훈제오리야채찜"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "매콤버섯볶음"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "병아리콩우엉조림"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "블루베리샐러드"
        },
        {
          "cornerName": "중식",
          "displayOrder": 7,
          "menuName": "배추김치"
        },
        {
          "cornerName": "중식",
          "displayOrder": 8,
          "menuName": "------------------"
        },
        {
          "cornerName": "중식",
          "displayOrder": 9,
          "menuName": "(11:00~11:40)"
        },
        {
          "cornerName": "중식",
          "displayOrder": 10,
          "menuName": "일품식당에서"
        },
        {
          "cornerName": "중식",
          "displayOrder": 11,
          "menuName": "위의 정식메뉴 운영"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "중식",
          "displayOrder": 1,
          "menuName": "기장밥"
        },
        {
          "cornerName": "중식",
          "displayOrder": 2,
          "menuName": "전복갈비탕"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "충무식오징어어묵무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "청포묵김가루무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "고추소박이"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "열대과일샐러드"
        },
        {
          "cornerName": "중식",
          "displayOrder": 7,
          "menuName": "깍두기"
        },
        {
          "cornerName": "중식",
          "displayOrder": 8,
          "menuName": "------------------"
        },
        {
          "cornerName": "중식",
          "displayOrder": 9,
          "menuName": "(11:00~11:40)"
        },
        {
          "cornerName": "중식",
          "displayOrder": 10,
          "menuName": "일품식당에서"
        },
        {
          "cornerName": "중식",
          "displayOrder": 11,
          "menuName": "위의 정식메뉴 운영"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
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
          "menuName": "북어해장국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "닭갈비순대볶음"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "이연복가지찜"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "콩나물파채무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "삼색쌈무"
        },
        {
          "cornerName": "중식",
          "displayOrder": 7,
          "menuName": "배추김치"
        },
        {
          "cornerName": "중식",
          "displayOrder": 8,
          "menuName": "------------------"
        },
        {
          "cornerName": "중식",
          "displayOrder": 9,
          "menuName": "(11:00~11:40)"
        },
        {
          "cornerName": "중식",
          "displayOrder": 10,
          "menuName": "일품식당에서"
        },
        {
          "cornerName": "중식",
          "displayOrder": 11,
          "menuName": "위의 정식메뉴 운영"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
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
          "menuName": "열무된장국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "어린잎순두부"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "고구마고로케"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "그래놀라샐러드"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "깍두기"
        },
        {
          "cornerName": "중식",
          "displayOrder": 7,
          "menuName": "------------------"
        },
        {
          "cornerName": "중식",
          "displayOrder": 8,
          "menuName": "(11:00~11:40)"
        },
        {
          "cornerName": "중식",
          "displayOrder": 9,
          "menuName": "일품식당에서"
        },
        {
          "cornerName": "중식",
          "displayOrder": 10,
          "menuName": "위의 정식메뉴 운영"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
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
          "menuName": "조랭이미역국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "옥수수계란찜"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "마늘쫑건새우볶음"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "빵가루마요샐러드"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "배추김치"
        },
        {
          "cornerName": "중식",
          "displayOrder": 7,
          "menuName": "------------------"
        },
        {
          "cornerName": "중식",
          "displayOrder": 8,
          "menuName": "(11:00~11:40)"
        },
        {
          "cornerName": "중식",
          "displayOrder": 9,
          "menuName": "일품식당에서"
        },
        {
          "cornerName": "중식",
          "displayOrder": 10,
          "menuName": "위의 정식메뉴 운영"
        }
      ]
    }
  ]
}
```

</details>

</details>

<details>
<summary><strong>분식당</strong> — HTTP 200, meals=10 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "금오공과대학교",
  "cafeteriaName": "분식당",
  "sourceUrl": "https://www.kumoh.ac.kr/ko/restaurant04.do",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04"
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
    "startDate": "2026-08-31",
    "endDate": "2026-09-04",
    "meals": [
      {
        "mealDate": "2026-08-31",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "일품식당에서 주문 가능"
          }
        ]
      },
      {
        "mealDate": "2026-08-31",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "일품식당에서 주문 가능"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "우동"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 2,
            "menuName": "떡만두라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 3,
            "menuName": "얼큰라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 4,
            "menuName": "치즈라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 5,
            "menuName": "라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 6,
            "menuName": "공깃밥"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 7,
            "menuName": "왕돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 8,
            "menuName": "고구마돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 9,
            "menuName": "치즈돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 10,
            "menuName": "닭강정"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "우동"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 2,
            "menuName": "떡만두라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 3,
            "menuName": "얼큰라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 4,
            "menuName": "치즈라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 5,
            "menuName": "라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 6,
            "menuName": "공깃밥"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 7,
            "menuName": "왕돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 8,
            "menuName": "고구마돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 9,
            "menuName": "치즈돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 10,
            "menuName": "닭강정"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "우동"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 2,
            "menuName": "떡만두라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 3,
            "menuName": "얼큰라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 4,
            "menuName": "치즈라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 5,
            "menuName": "라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 6,
            "menuName": "공깃밥"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 7,
            "menuName": "왕돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 8,
            "menuName": "고구마돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 9,
            "menuName": "치즈돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 10,
            "menuName": "닭강정"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "우동"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 2,
            "menuName": "떡만두라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 3,
            "menuName": "얼큰라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 4,
            "menuName": "치즈라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 5,
            "menuName": "라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 6,
            "menuName": "공깃밥"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 7,
            "menuName": "왕돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 8,
            "menuName": "고구마돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 9,
            "menuName": "치즈돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 10,
            "menuName": "닭강정"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "우동"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 2,
            "menuName": "떡만두라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 3,
            "menuName": "얼큰라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 4,
            "menuName": "치즈라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 5,
            "menuName": "라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 6,
            "menuName": "공깃밥"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 7,
            "menuName": "왕돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 8,
            "menuName": "고구마돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 9,
            "menuName": "치즈돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 10,
            "menuName": "닭강정"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "우동"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 2,
            "menuName": "떡만두라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 3,
            "menuName": "얼큰라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 4,
            "menuName": "치즈라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 5,
            "menuName": "라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 6,
            "menuName": "공깃밥"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 7,
            "menuName": "왕돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 8,
            "menuName": "고구마돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 9,
            "menuName": "치즈돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 10,
            "menuName": "닭강정"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "우동"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 2,
            "menuName": "떡만두라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 3,
            "menuName": "얼큰라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 4,
            "menuName": "치즈라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 5,
            "menuName": "라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 6,
            "menuName": "공깃밥"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 7,
            "menuName": "왕돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 8,
            "menuName": "고구마돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 9,
            "menuName": "치즈돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 10,
            "menuName": "닭강정"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "일품요리",
            "displayOrder": 1,
            "menuName": "우동"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 2,
            "menuName": "떡만두라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 3,
            "menuName": "얼큰라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 4,
            "menuName": "치즈라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 5,
            "menuName": "라면"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 6,
            "menuName": "공깃밥"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 7,
            "menuName": "왕돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 8,
            "menuName": "고구마돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 9,
            "menuName": "치즈돈가스"
          },
          {
            "cornerName": "일품요리",
            "displayOrder": 10,
            "menuName": "닭강정"
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
  "startDate": "2026-08-31",
  "endDate": "2026-09-04",
  "meals": [
    {
      "mealDate": "2026-08-31",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "일품식당에서 주문 가능"
        }
      ]
    },
    {
      "mealDate": "2026-08-31",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "일품식당에서 주문 가능"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "우동"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 2,
          "menuName": "떡만두라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 3,
          "menuName": "얼큰라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 4,
          "menuName": "치즈라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 5,
          "menuName": "라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 6,
          "menuName": "공깃밥"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 7,
          "menuName": "왕돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 8,
          "menuName": "고구마돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 9,
          "menuName": "치즈돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 10,
          "menuName": "닭강정"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "우동"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 2,
          "menuName": "떡만두라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 3,
          "menuName": "얼큰라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 4,
          "menuName": "치즈라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 5,
          "menuName": "라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 6,
          "menuName": "공깃밥"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 7,
          "menuName": "왕돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 8,
          "menuName": "고구마돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 9,
          "menuName": "치즈돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 10,
          "menuName": "닭강정"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "우동"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 2,
          "menuName": "떡만두라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 3,
          "menuName": "얼큰라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 4,
          "menuName": "치즈라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 5,
          "menuName": "라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 6,
          "menuName": "공깃밥"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 7,
          "menuName": "왕돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 8,
          "menuName": "고구마돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 9,
          "menuName": "치즈돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 10,
          "menuName": "닭강정"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "우동"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 2,
          "menuName": "떡만두라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 3,
          "menuName": "얼큰라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 4,
          "menuName": "치즈라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 5,
          "menuName": "라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 6,
          "menuName": "공깃밥"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 7,
          "menuName": "왕돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 8,
          "menuName": "고구마돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 9,
          "menuName": "치즈돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 10,
          "menuName": "닭강정"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "우동"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 2,
          "menuName": "떡만두라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 3,
          "menuName": "얼큰라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 4,
          "menuName": "치즈라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 5,
          "menuName": "라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 6,
          "menuName": "공깃밥"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 7,
          "menuName": "왕돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 8,
          "menuName": "고구마돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 9,
          "menuName": "치즈돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 10,
          "menuName": "닭강정"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "우동"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 2,
          "menuName": "떡만두라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 3,
          "menuName": "얼큰라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 4,
          "menuName": "치즈라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 5,
          "menuName": "라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 6,
          "menuName": "공깃밥"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 7,
          "menuName": "왕돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 8,
          "menuName": "고구마돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 9,
          "menuName": "치즈돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 10,
          "menuName": "닭강정"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "우동"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 2,
          "menuName": "떡만두라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 3,
          "menuName": "얼큰라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 4,
          "menuName": "치즈라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 5,
          "menuName": "라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 6,
          "menuName": "공깃밥"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 7,
          "menuName": "왕돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 8,
          "menuName": "고구마돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 9,
          "menuName": "치즈돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 10,
          "menuName": "닭강정"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "일품요리",
          "displayOrder": 1,
          "menuName": "우동"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 2,
          "menuName": "떡만두라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 3,
          "menuName": "얼큰라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 4,
          "menuName": "치즈라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 5,
          "menuName": "라면"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 6,
          "menuName": "공깃밥"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 7,
          "menuName": "왕돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 8,
          "menuName": "고구마돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 9,
          "menuName": "치즈돈가스"
        },
        {
          "cornerName": "일품요리",
          "displayOrder": 10,
          "menuName": "닭강정"
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
<summary><strong>정보센터식당</strong> — HTTP 200, meals=10 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "정보센터식당",
  "sourceUrl": "https://coop.knu.ac.kr/?shop_sqno=35",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "경북대학교",
    "cafeteriaName": "정보센터식당",
    "sourceUrl": "https://coop.knu.ac.kr/?shop_sqno=35",
    "startDate": "2026-08-31",
    "endDate": "2026-09-04",
    "meals": [
      {
        "mealDate": "2026-08-31",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "바베큐폭립오므라이스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 2,
            "menuName": "육전비빔국수★ 망고쥬스"
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
        "mealDate": "2026-08-31",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "1식4찬 자율배식"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "돈코츠라멘★ 소떡소떡★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 2,
            "menuName": "콩불냄비★ 갈비만두★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 3,
            "menuName": "육회비빔밥"
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
        "mealDate": "2026-09-01",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "1식4찬 자율배식"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "동인동찜갈비★ 참치마요"
          },
          {
            "cornerName": "특식",
            "displayOrder": 2,
            "menuName": "육회불닭냉면"
          },
          {
            "cornerName": "특식",
            "displayOrder": 3,
            "menuName": "닭가슴살비빔밥"
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
        "mealDate": "2026-09-02",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "1식4찬 자율배식"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "불닭크런치야끼우동"
          },
          {
            "cornerName": "특식",
            "displayOrder": 2,
            "menuName": "누룽지백숙"
          },
          {
            "cornerName": "특식",
            "displayOrder": 3,
            "menuName": "오삼비빔밥★"
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
        "mealDate": "2026-09-03",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "1식4찬 자율배식"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "왕새우메밀소바 대왕유부초밥"
          },
          {
            "cornerName": "특식",
            "displayOrder": 2,
            "menuName": "돈육불고기덮밥★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 3,
            "menuName": "제육치즈돌솥★"
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
        "mealDate": "2026-09-04",
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
  "sourceUrl": "https://coop.knu.ac.kr/?shop_sqno=35",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04",
  "meals": [
    {
      "mealDate": "2026-08-31",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "바베큐폭립오므라이스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 2,
          "menuName": "육전비빔국수★ 망고쥬스"
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
      "mealDate": "2026-08-31",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "1식4찬 자율배식"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "돈코츠라멘★ 소떡소떡★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 2,
          "menuName": "콩불냄비★ 갈비만두★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 3,
          "menuName": "육회비빔밥"
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
      "mealDate": "2026-09-01",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "1식4찬 자율배식"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "동인동찜갈비★ 참치마요"
        },
        {
          "cornerName": "특식",
          "displayOrder": 2,
          "menuName": "육회불닭냉면"
        },
        {
          "cornerName": "특식",
          "displayOrder": 3,
          "menuName": "닭가슴살비빔밥"
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
      "mealDate": "2026-09-02",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "1식4찬 자율배식"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "불닭크런치야끼우동"
        },
        {
          "cornerName": "특식",
          "displayOrder": 2,
          "menuName": "누룽지백숙"
        },
        {
          "cornerName": "특식",
          "displayOrder": 3,
          "menuName": "오삼비빔밥★"
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
      "mealDate": "2026-09-03",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "1식4찬 자율배식"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "왕새우메밀소바 대왕유부초밥"
        },
        {
          "cornerName": "특식",
          "displayOrder": 2,
          "menuName": "돈육불고기덮밥★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 3,
          "menuName": "제육치즈돌솥★"
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
      "mealDate": "2026-09-04",
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
<summary><strong>복지관 교직원식당</strong> — HTTP 200, meals=5 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "복지관 교직원식당",
  "sourceUrl": "https://coop.knu.ac.kr/?shop_sqno=36",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "경북대학교",
    "cafeteriaName": "복지관 교직원식당",
    "sourceUrl": "https://coop.knu.ac.kr/?shop_sqno=36",
    "startDate": "2026-08-31",
    "endDate": "2026-09-04",
    "meals": [
      {
        "mealDate": "2026-08-31",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "귀리밥 북어계란국 치즈콩불★ 백순대들깨볶음★ 오이생크림무침 포기김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "흰밥 유부김치국 뿌링클치킨& 요거트소스 볼로네제파스타★ 그래놀라샐러드 깍두기"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "*생일밥상* 찰밥 새알미역국 고추장삼겹살★ 한식잡채★ 흑임자샐러드 팩주스(오렌지,망고) 포기김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "후리가케밥 냉메밀소바 매콤찜닭 왕새우튀김& 허니머스타드 열대과일샐러드 포기김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "흰밥 두부된장국 소불고기 옹심이고기조림★ 김가루비빔면 포기김치"
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
  "sourceUrl": "https://coop.knu.ac.kr/?shop_sqno=36",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04",
  "meals": [
    {
      "mealDate": "2026-08-31",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "귀리밥 북어계란국 치즈콩불★ 백순대들깨볶음★ 오이생크림무침 포기김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "흰밥 유부김치국 뿌링클치킨& 요거트소스 볼로네제파스타★ 그래놀라샐러드 깍두기"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "*생일밥상* 찰밥 새알미역국 고추장삼겹살★ 한식잡채★ 흑임자샐러드 팩주스(오렌지,망고) 포기김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "후리가케밥 냉메밀소바 매콤찜닭 왕새우튀김& 허니머스타드 열대과일샐러드 포기김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "흰밥 두부된장국 소불고기 옹심이고기조림★ 김가루비빔면 포기김치"
        }
      ]
    }
  ]
}
```

</details>

</details>

<details>
<summary><strong>카페테리아 첨성</strong> — HTTP 200, meals=10 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "카페테리아 첨성",
  "sourceUrl": "https://coop.knu.ac.kr/?shop_sqno=37",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "경북대학교",
    "cafeteriaName": "카페테리아 첨성",
    "sourceUrl": "https://coop.knu.ac.kr/?shop_sqno=37",
    "startDate": "2026-08-31",
    "endDate": "2026-09-04",
    "meals": [
      {
        "mealDate": "2026-08-31",
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
        "mealDate": "2026-08-31",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "팟타이★ 스프링롤샐러드"
          },
          {
            "cornerName": "특식",
            "displayOrder": 2,
            "menuName": "소시지 오므라이스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 3,
            "menuName": "참치비빔밥"
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
            "menuName": "치즈왕돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 9,
            "menuName": "라면"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
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
        "mealDate": "2026-09-01",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "돈목살 스테이크★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 2,
            "menuName": "더진국 수육국밥★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 3,
            "menuName": "더진국 순대수육국밥★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 4,
            "menuName": "육회비빔밥"
          },
          {
            "cornerName": "특식",
            "displayOrder": 5,
            "menuName": "순살돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 6,
            "menuName": "치즈돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 7,
            "menuName": "고구마돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 8,
            "menuName": "왕돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 9,
            "menuName": "치즈왕돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 10,
            "menuName": "라면"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
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
        "mealDate": "2026-09-02",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "알리고치즈감자 바베큐★ &크로와상 &샐러드"
          },
          {
            "cornerName": "특식",
            "displayOrder": 2,
            "menuName": "더진국 수육국밥★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 3,
            "menuName": "더진국 순대수육국밥★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 4,
            "menuName": "불고기비빔밥★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 5,
            "menuName": "순살돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 6,
            "menuName": "치즈돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 7,
            "menuName": "고구마돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 8,
            "menuName": "왕돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 9,
            "menuName": "치즈왕돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 10,
            "menuName": "라면"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
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
        "mealDate": "2026-09-03",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "중국식계란볶음밥& 해물짬뽕면★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 2,
            "menuName": "더진국 수육국밥★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 3,
            "menuName": "더진국 순대수육국밥★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 4,
            "menuName": "닭불고기비빔밥"
          },
          {
            "cornerName": "특식",
            "displayOrder": 5,
            "menuName": "순살돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 6,
            "menuName": "치즈돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 7,
            "menuName": "고구마돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 8,
            "menuName": "왕돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 9,
            "menuName": "치즈왕돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 10,
            "menuName": "라면"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
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
        "mealDate": "2026-09-04",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "특식",
            "displayOrder": 1,
            "menuName": "불고기파스타& 마늘빵"
          },
          {
            "cornerName": "특식",
            "displayOrder": 2,
            "menuName": "더진국 수육국밥★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 3,
            "menuName": "더진국 순대수육국밥★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 4,
            "menuName": "오삼비빔밥★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 5,
            "menuName": "순살돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 6,
            "menuName": "치즈돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 7,
            "menuName": "고구마돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 8,
            "menuName": "왕돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 9,
            "menuName": "치즈왕돈가스★"
          },
          {
            "cornerName": "특식",
            "displayOrder": 10,
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
  "sourceUrl": "https://coop.knu.ac.kr/?shop_sqno=37",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04",
  "meals": [
    {
      "mealDate": "2026-08-31",
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
      "mealDate": "2026-08-31",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "팟타이★ 스프링롤샐러드"
        },
        {
          "cornerName": "특식",
          "displayOrder": 2,
          "menuName": "소시지 오므라이스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 3,
          "menuName": "참치비빔밥"
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
          "menuName": "치즈왕돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 9,
          "menuName": "라면"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
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
      "mealDate": "2026-09-01",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "돈목살 스테이크★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 2,
          "menuName": "더진국 수육국밥★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 3,
          "menuName": "더진국 순대수육국밥★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 4,
          "menuName": "육회비빔밥"
        },
        {
          "cornerName": "특식",
          "displayOrder": 5,
          "menuName": "순살돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 6,
          "menuName": "치즈돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 7,
          "menuName": "고구마돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 8,
          "menuName": "왕돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 9,
          "menuName": "치즈왕돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 10,
          "menuName": "라면"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
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
      "mealDate": "2026-09-02",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "알리고치즈감자 바베큐★ &크로와상 &샐러드"
        },
        {
          "cornerName": "특식",
          "displayOrder": 2,
          "menuName": "더진국 수육국밥★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 3,
          "menuName": "더진국 순대수육국밥★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 4,
          "menuName": "불고기비빔밥★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 5,
          "menuName": "순살돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 6,
          "menuName": "치즈돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 7,
          "menuName": "고구마돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 8,
          "menuName": "왕돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 9,
          "menuName": "치즈왕돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 10,
          "menuName": "라면"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
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
      "mealDate": "2026-09-03",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "중국식계란볶음밥& 해물짬뽕면★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 2,
          "menuName": "더진국 수육국밥★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 3,
          "menuName": "더진국 순대수육국밥★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 4,
          "menuName": "닭불고기비빔밥"
        },
        {
          "cornerName": "특식",
          "displayOrder": 5,
          "menuName": "순살돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 6,
          "menuName": "치즈돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 7,
          "menuName": "고구마돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 8,
          "menuName": "왕돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 9,
          "menuName": "치즈왕돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 10,
          "menuName": "라면"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
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
      "mealDate": "2026-09-04",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "특식",
          "displayOrder": 1,
          "menuName": "불고기파스타& 마늘빵"
        },
        {
          "cornerName": "특식",
          "displayOrder": 2,
          "menuName": "더진국 수육국밥★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 3,
          "menuName": "더진국 순대수육국밥★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 4,
          "menuName": "오삼비빔밥★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 5,
          "menuName": "순살돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 6,
          "menuName": "치즈돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 7,
          "menuName": "고구마돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 8,
          "menuName": "왕돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 9,
          "menuName": "치즈왕돈가스★"
        },
        {
          "cornerName": "특식",
          "displayOrder": 10,
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
<summary><strong>GP감꽃식당</strong> — HTTP 200, meals=5 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "GP감꽃식당",
  "sourceUrl": "https://coop.knu.ac.kr/?shop_sqno=46",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "경북대학교",
    "cafeteriaName": "GP감꽃식당",
    "sourceUrl": "https://coop.knu.ac.kr/?shop_sqno=46",
    "startDate": "2026-08-31",
    "endDate": "2026-09-04",
    "meals": [
      {
        "mealDate": "2026-08-31",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "백미밥 계란파개장 닭볶음탕 돈채마파가지볶음★ 콩나물무침 포기김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "백미밥 돼지고기두부찌개★ 소세지카레★ 모듬튀김&소스★ 얼갈이무침/오이무침 포기김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "<생일밥상> 찰밥 새알미역국 고추장삼겹살★ 한식잡채 흑임자샐러드 팩주스/포기김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "현미밥 오징어무국 순살로제찜닭 떡갈비&마늘소스★ 파프리카두부무침 매실차/포기김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "백미밥 참깨만두국★ 김치제육볶음★ 소떡소떡★ 청경채깨무침 열무김치"
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
  "sourceUrl": "https://coop.knu.ac.kr/?shop_sqno=46",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04",
  "meals": [
    {
      "mealDate": "2026-08-31",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "백미밥 계란파개장 닭볶음탕 돈채마파가지볶음★ 콩나물무침 포기김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "백미밥 돼지고기두부찌개★ 소세지카레★ 모듬튀김&소스★ 얼갈이무침/오이무침 포기김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "<생일밥상> 찰밥 새알미역국 고추장삼겹살★ 한식잡채 흑임자샐러드 팩주스/포기김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "현미밥 오징어무국 순살로제찜닭 떡갈비&마늘소스★ 파프리카두부무침 매실차/포기김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "백미밥 참깨만두국★ 김치제육볶음★ 소떡소떡★ 청경채깨무침 열무김치"
        }
      ]
    }
  ]
}
```

</details>

</details>

<details>
<summary><strong>공학관교직원식당(외부업체)</strong> — HTTP 200, meals=10 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "경북대학교",
  "cafeteriaName": "공학관교직원식당(외부업체)",
  "sourceUrl": "https://coop.knu.ac.kr/?shop_sqno=85",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "경북대학교",
    "cafeteriaName": "공학관교직원식당(외부업체)",
    "sourceUrl": "https://coop.knu.ac.kr/?shop_sqno=85",
    "startDate": "2026-08-31",
    "endDate": "2026-09-04",
    "meals": [
      {
        "mealDate": "2026-08-31",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "흑미밥/쌀밥 재첩국 안동식매콤찜닭 야채튀김&초간장 열무된장무침 직접담근김치 잔치국수"
          }
        ]
      },
      {
        "mealDate": "2026-08-31",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "기장밥/쌀밥 부대찌개★ 꿔바로우★ 양파초절임 직접담근김치 도시락김"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "기장밥/쌀밥 미더덕된장찌개 묵은지돈육찜★ 새우볼튀김&칠리소스 단배추겉절이 무우말랭이 잔치국수"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "흑미밥/쌀밥 피홍합탕 케이준치킨샐러드 햄계란구이&케찹★ 영양콩조림 직접담근김치 도시락김"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "흑미밥/쌀밥 북어채해장국 부추떡갈비★ 계란말이 청경채겉절이 직접담근김치 잔치국수"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "기장밥/쌀밥 미니우동국 홍초불닭 알감자조림 느타리부추무침 직접담근김치 도시락김"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "기장밥/쌀밥 경상도식소고기국 인절미탕수육★ 수제비어묵매콤볶음 쑥갓두부무침 직접담근김치 잔치국수"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "흑미밥/쌀밥 어묵무국 치킨너겟데리야끼강정 나폴리탄파스타★ 브로컬리숙회&초고추장 직접담근김치 도시락김"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "흑미밥/쌀밥 나가사끼짬뽕국★ 탄두리치킨&요구르트드레싱 찐만두&초간장★ 오이송송이 직접담근김치 잔치국수"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "정식",
            "displayOrder": 1,
            "menuName": "기장밥/쌀밥 도토리묵국 돼지고기파채불고기★ 오징어튀김&초간장 연근채흑임자소스무침 직접담근김치 도시락김"
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
  "sourceUrl": "https://coop.knu.ac.kr/?shop_sqno=85",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04",
  "meals": [
    {
      "mealDate": "2026-08-31",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "흑미밥/쌀밥 재첩국 안동식매콤찜닭 야채튀김&초간장 열무된장무침 직접담근김치 잔치국수"
        }
      ]
    },
    {
      "mealDate": "2026-08-31",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "기장밥/쌀밥 부대찌개★ 꿔바로우★ 양파초절임 직접담근김치 도시락김"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "기장밥/쌀밥 미더덕된장찌개 묵은지돈육찜★ 새우볼튀김&칠리소스 단배추겉절이 무우말랭이 잔치국수"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "흑미밥/쌀밥 피홍합탕 케이준치킨샐러드 햄계란구이&케찹★ 영양콩조림 직접담근김치 도시락김"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "흑미밥/쌀밥 북어채해장국 부추떡갈비★ 계란말이 청경채겉절이 직접담근김치 잔치국수"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "기장밥/쌀밥 미니우동국 홍초불닭 알감자조림 느타리부추무침 직접담근김치 도시락김"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "기장밥/쌀밥 경상도식소고기국 인절미탕수육★ 수제비어묵매콤볶음 쑥갓두부무침 직접담근김치 잔치국수"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "흑미밥/쌀밥 어묵무국 치킨너겟데리야끼강정 나폴리탄파스타★ 브로컬리숙회&초고추장 직접담근김치 도시락김"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "흑미밥/쌀밥 나가사끼짬뽕국★ 탄두리치킨&요구르트드레싱 찐만두&초간장★ 오이송송이 직접담근김치 잔치국수"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "정식",
          "displayOrder": 1,
          "menuName": "기장밥/쌀밥 도토리묵국 돼지고기파채불고기★ 오징어튀김&초간장 연근채흑임자소스무침 직접담근김치 도시락김"
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
<summary><strong>이룸관(안동, 학생식당)</strong> — HTTP 200, meals=9 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "국립경국대학교",
  "cafeteriaName": "이룸관(안동, 학생식당)",
  "sourceUrl": "https://www.gknu.ac.kr/?menu_idx=82",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "국립경국대학교",
    "cafeteriaName": "이룸관(안동, 학생식당)",
    "sourceUrl": "https://www.gknu.ac.kr/?menu_idx=82",
    "startDate": "2026-08-31",
    "endDate": "2026-09-04",
    "meals": [
      {
        "mealDate": "2026-08-31",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "중식",
            "displayOrder": 1,
            "menuName": "※ 천원의 브런치 : 재학생 키오스크 천원 식권 발급 후 모바일 학생증 제시"
          },
          {
            "cornerName": "중식",
            "displayOrder": 2,
            "menuName": "흑미밥"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "골부리국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "닭양념오븐구이"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "감자양파매운볶음"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "우엉조림"
          },
          {
            "cornerName": "중식",
            "displayOrder": 7,
            "menuName": "청경채겉절이"
          },
          {
            "cornerName": "중식",
            "displayOrder": 8,
            "menuName": "배추김치/요구르트"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
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
            "menuName": "얼큰애호박국"
          },
          {
            "cornerName": "조식",
            "displayOrder": 4,
            "menuName": "닭갈비"
          },
          {
            "cornerName": "조식",
            "displayOrder": 5,
            "menuName": "고추튀김"
          },
          {
            "cornerName": "조식",
            "displayOrder": 6,
            "menuName": "콩나물무침"
          },
          {
            "cornerName": "조식",
            "displayOrder": 7,
            "menuName": "배추김치/마시는요거트"
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
        "mealDate": "2026-09-01",
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
            "menuName": "꽃게된장국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "돼지갈비찜"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "과일샐러드"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "열무무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "김구이"
          },
          {
            "cornerName": "중식",
            "displayOrder": 7,
            "menuName": "배추김치/요구르트"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
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
            "menuName": "골부리국"
          },
          {
            "cornerName": "조식",
            "displayOrder": 4,
            "menuName": "소불고기"
          },
          {
            "cornerName": "조식",
            "displayOrder": 5,
            "menuName": "계란야채찜"
          },
          {
            "cornerName": "조식",
            "displayOrder": 6,
            "menuName": "꼬시래기무침"
          },
          {
            "cornerName": "조식",
            "displayOrder": 7,
            "menuName": "배추김치/감귤과채소쥬스"
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
            "menuName": "감귤과채쥬스"
          },
          {
            "cornerName": "조식",
            "displayOrder": 11,
            "menuName": "바나나"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
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
            "menuName": "느타리버섯국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "닭강정"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "감자샐러드"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "노가리풋고추무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "마파두부"
          },
          {
            "cornerName": "중식",
            "displayOrder": 7,
            "menuName": "배추김치/단호박죽"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
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
            "menuName": "참치김치찌개"
          },
          {
            "cornerName": "조식",
            "displayOrder": 4,
            "menuName": "닭양념오븐구이"
          },
          {
            "cornerName": "조식",
            "displayOrder": 5,
            "menuName": "감자양파매운볶음"
          },
          {
            "cornerName": "조식",
            "displayOrder": 6,
            "menuName": "열무된장무침"
          },
          {
            "cornerName": "조식",
            "displayOrder": 7,
            "menuName": "배추김치/요구르트"
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
            "menuName": "비요뜨"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
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
            "menuName": "닭개장"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "갈치무조림"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "깐풍만두"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "해물콩나물찜"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "얼갈이겉절이"
          },
          {
            "cornerName": "중식",
            "displayOrder": 7,
            "menuName": "배추김치/요구르트"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
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
            "menuName": "쇠고기무국"
          },
          {
            "cornerName": "조식",
            "displayOrder": 4,
            "menuName": "돼지가지볶음"
          },
          {
            "cornerName": "조식",
            "displayOrder": 5,
            "menuName": "두부조림"
          },
          {
            "cornerName": "조식",
            "displayOrder": 6,
            "menuName": "콩나물무침"
          },
          {
            "cornerName": "조식",
            "displayOrder": 7,
            "menuName": "배추김치/얼라이브쥬스"
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
            "menuName": "얼라이브쥬스"
          },
          {
            "cornerName": "조식",
            "displayOrder": 11,
            "menuName": "바나나"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "중식",
            "displayOrder": 1,
            "menuName": "도토리묵비빔밥&장볶이"
          },
          {
            "cornerName": "중식",
            "displayOrder": 2,
            "menuName": "유부국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "꿔바로우"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "오이양파무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "깍두기"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "배추김치/마시는 요거트"
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
  "sourceUrl": "https://www.gknu.ac.kr/?menu_idx=82",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04",
  "meals": [
    {
      "mealDate": "2026-08-31",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "중식",
          "displayOrder": 1,
          "menuName": "※ 천원의 브런치 : 재학생 키오스크 천원 식권 발급 후 모바일 학생증 제시"
        },
        {
          "cornerName": "중식",
          "displayOrder": 2,
          "menuName": "흑미밥"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "골부리국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "닭양념오븐구이"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "감자양파매운볶음"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "우엉조림"
        },
        {
          "cornerName": "중식",
          "displayOrder": 7,
          "menuName": "청경채겉절이"
        },
        {
          "cornerName": "중식",
          "displayOrder": 8,
          "menuName": "배추김치/요구르트"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
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
          "menuName": "얼큰애호박국"
        },
        {
          "cornerName": "조식",
          "displayOrder": 4,
          "menuName": "닭갈비"
        },
        {
          "cornerName": "조식",
          "displayOrder": 5,
          "menuName": "고추튀김"
        },
        {
          "cornerName": "조식",
          "displayOrder": 6,
          "menuName": "콩나물무침"
        },
        {
          "cornerName": "조식",
          "displayOrder": 7,
          "menuName": "배추김치/마시는요거트"
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
      "mealDate": "2026-09-01",
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
          "menuName": "꽃게된장국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "돼지갈비찜"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "과일샐러드"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "열무무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "김구이"
        },
        {
          "cornerName": "중식",
          "displayOrder": 7,
          "menuName": "배추김치/요구르트"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
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
          "menuName": "골부리국"
        },
        {
          "cornerName": "조식",
          "displayOrder": 4,
          "menuName": "소불고기"
        },
        {
          "cornerName": "조식",
          "displayOrder": 5,
          "menuName": "계란야채찜"
        },
        {
          "cornerName": "조식",
          "displayOrder": 6,
          "menuName": "꼬시래기무침"
        },
        {
          "cornerName": "조식",
          "displayOrder": 7,
          "menuName": "배추김치/감귤과채소쥬스"
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
          "menuName": "감귤과채쥬스"
        },
        {
          "cornerName": "조식",
          "displayOrder": 11,
          "menuName": "바나나"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
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
          "menuName": "느타리버섯국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "닭강정"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "감자샐러드"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "노가리풋고추무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "마파두부"
        },
        {
          "cornerName": "중식",
          "displayOrder": 7,
          "menuName": "배추김치/단호박죽"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
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
          "menuName": "참치김치찌개"
        },
        {
          "cornerName": "조식",
          "displayOrder": 4,
          "menuName": "닭양념오븐구이"
        },
        {
          "cornerName": "조식",
          "displayOrder": 5,
          "menuName": "감자양파매운볶음"
        },
        {
          "cornerName": "조식",
          "displayOrder": 6,
          "menuName": "열무된장무침"
        },
        {
          "cornerName": "조식",
          "displayOrder": 7,
          "menuName": "배추김치/요구르트"
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
          "menuName": "비요뜨"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
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
          "menuName": "닭개장"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "갈치무조림"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "깐풍만두"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "해물콩나물찜"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "얼갈이겉절이"
        },
        {
          "cornerName": "중식",
          "displayOrder": 7,
          "menuName": "배추김치/요구르트"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
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
          "menuName": "쇠고기무국"
        },
        {
          "cornerName": "조식",
          "displayOrder": 4,
          "menuName": "돼지가지볶음"
        },
        {
          "cornerName": "조식",
          "displayOrder": 5,
          "menuName": "두부조림"
        },
        {
          "cornerName": "조식",
          "displayOrder": 6,
          "menuName": "콩나물무침"
        },
        {
          "cornerName": "조식",
          "displayOrder": 7,
          "menuName": "배추김치/얼라이브쥬스"
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
          "menuName": "얼라이브쥬스"
        },
        {
          "cornerName": "조식",
          "displayOrder": 11,
          "menuName": "바나나"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "중식",
          "displayOrder": 1,
          "menuName": "도토리묵비빔밥&장볶이"
        },
        {
          "cornerName": "중식",
          "displayOrder": 2,
          "menuName": "유부국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "꿔바로우"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "오이양파무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "깍두기"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "배추김치/마시는 요거트"
        }
      ]
    }
  ]
}
```

</details>

</details>

<details>
<summary><strong>채움관(안동, 교직원식당)</strong> — HTTP 200, meals=9 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "국립경국대학교",
  "cafeteriaName": "채움관(안동, 교직원식당)",
  "sourceUrl": "https://www.gknu.ac.kr/?menu_idx=222",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "국립경국대학교",
    "cafeteriaName": "채움관(안동, 교직원식당)",
    "sourceUrl": "https://www.gknu.ac.kr/?menu_idx=222",
    "startDate": "2026-08-31",
    "endDate": "2026-09-04",
    "meals": [
      {
        "mealDate": "2026-08-31",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "중식",
            "displayOrder": 1,
            "menuName": "※ 천원의 브런치 : 재학생 키오스크 천원 식권 발급 후 모바일 학생증 제시"
          },
          {
            "cornerName": "중식",
            "displayOrder": 2,
            "menuName": "흑미밥"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "골부리국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "닭양념오븐구이"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "감자양파매운볶음"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "우엉조림"
          },
          {
            "cornerName": "중식",
            "displayOrder": 7,
            "menuName": "청경채겉절이"
          },
          {
            "cornerName": "중식",
            "displayOrder": 8,
            "menuName": "배추김치/요구르트"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
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
            "menuName": "얼큰애호박국"
          },
          {
            "cornerName": "조식",
            "displayOrder": 4,
            "menuName": "닭갈비"
          },
          {
            "cornerName": "조식",
            "displayOrder": 5,
            "menuName": "고추튀김"
          },
          {
            "cornerName": "조식",
            "displayOrder": 6,
            "menuName": "콩나물무침"
          },
          {
            "cornerName": "조식",
            "displayOrder": 7,
            "menuName": "배추김치/마시는요거트"
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
        "mealDate": "2026-09-01",
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
            "menuName": "꽃게된장국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "돼지갈비찜"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "과일샐러드"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "열무무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "김구이"
          },
          {
            "cornerName": "중식",
            "displayOrder": 7,
            "menuName": "배추김치/요구르트"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
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
            "menuName": "골부리국"
          },
          {
            "cornerName": "조식",
            "displayOrder": 4,
            "menuName": "소불고기"
          },
          {
            "cornerName": "조식",
            "displayOrder": 5,
            "menuName": "계란야채찜"
          },
          {
            "cornerName": "조식",
            "displayOrder": 6,
            "menuName": "꼬시래기무침"
          },
          {
            "cornerName": "조식",
            "displayOrder": 7,
            "menuName": "배추김치/감귤과채소쥬스"
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
            "menuName": "감귤과채쥬스"
          },
          {
            "cornerName": "조식",
            "displayOrder": 11,
            "menuName": "바나나"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
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
            "menuName": "느타리버섯국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "닭강정"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "감자샐러드"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "노가리풋고추무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "마파두부"
          },
          {
            "cornerName": "중식",
            "displayOrder": 7,
            "menuName": "배추김치/단호박죽"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
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
            "menuName": "참치김치찌개"
          },
          {
            "cornerName": "조식",
            "displayOrder": 4,
            "menuName": "닭양념오븐구이"
          },
          {
            "cornerName": "조식",
            "displayOrder": 5,
            "menuName": "감자양파매운볶음"
          },
          {
            "cornerName": "조식",
            "displayOrder": 6,
            "menuName": "열무된장무침"
          },
          {
            "cornerName": "조식",
            "displayOrder": 7,
            "menuName": "배추김치/요구르트"
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
            "menuName": "비요뜨"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
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
            "menuName": "닭개장"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "갈치무조림"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "깐풍만두"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "해물콩나물찜"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "얼갈이겉절이"
          },
          {
            "cornerName": "중식",
            "displayOrder": 7,
            "menuName": "배추김치/요구르트"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
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
            "menuName": "쇠고기무국"
          },
          {
            "cornerName": "조식",
            "displayOrder": 4,
            "menuName": "돼지가지볶음"
          },
          {
            "cornerName": "조식",
            "displayOrder": 5,
            "menuName": "두부조림"
          },
          {
            "cornerName": "조식",
            "displayOrder": 6,
            "menuName": "콩나물무침"
          },
          {
            "cornerName": "조식",
            "displayOrder": 7,
            "menuName": "배추김치/얼라이브쥬스"
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
            "menuName": "얼라이브쥬스"
          },
          {
            "cornerName": "조식",
            "displayOrder": 11,
            "menuName": "바나나"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "중식",
            "displayOrder": 1,
            "menuName": "도토리묵비빔밥&장볶이"
          },
          {
            "cornerName": "중식",
            "displayOrder": 2,
            "menuName": "유부국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "꿔바로우"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "오이양파무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "깍두기"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "배추김치/마시는 요거트"
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
  "sourceUrl": "https://www.gknu.ac.kr/?menu_idx=222",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04",
  "meals": [
    {
      "mealDate": "2026-08-31",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "중식",
          "displayOrder": 1,
          "menuName": "※ 천원의 브런치 : 재학생 키오스크 천원 식권 발급 후 모바일 학생증 제시"
        },
        {
          "cornerName": "중식",
          "displayOrder": 2,
          "menuName": "흑미밥"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "골부리국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "닭양념오븐구이"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "감자양파매운볶음"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "우엉조림"
        },
        {
          "cornerName": "중식",
          "displayOrder": 7,
          "menuName": "청경채겉절이"
        },
        {
          "cornerName": "중식",
          "displayOrder": 8,
          "menuName": "배추김치/요구르트"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
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
          "menuName": "얼큰애호박국"
        },
        {
          "cornerName": "조식",
          "displayOrder": 4,
          "menuName": "닭갈비"
        },
        {
          "cornerName": "조식",
          "displayOrder": 5,
          "menuName": "고추튀김"
        },
        {
          "cornerName": "조식",
          "displayOrder": 6,
          "menuName": "콩나물무침"
        },
        {
          "cornerName": "조식",
          "displayOrder": 7,
          "menuName": "배추김치/마시는요거트"
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
      "mealDate": "2026-09-01",
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
          "menuName": "꽃게된장국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "돼지갈비찜"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "과일샐러드"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "열무무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "김구이"
        },
        {
          "cornerName": "중식",
          "displayOrder": 7,
          "menuName": "배추김치/요구르트"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
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
          "menuName": "골부리국"
        },
        {
          "cornerName": "조식",
          "displayOrder": 4,
          "menuName": "소불고기"
        },
        {
          "cornerName": "조식",
          "displayOrder": 5,
          "menuName": "계란야채찜"
        },
        {
          "cornerName": "조식",
          "displayOrder": 6,
          "menuName": "꼬시래기무침"
        },
        {
          "cornerName": "조식",
          "displayOrder": 7,
          "menuName": "배추김치/감귤과채소쥬스"
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
          "menuName": "감귤과채쥬스"
        },
        {
          "cornerName": "조식",
          "displayOrder": 11,
          "menuName": "바나나"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
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
          "menuName": "느타리버섯국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "닭강정"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "감자샐러드"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "노가리풋고추무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "마파두부"
        },
        {
          "cornerName": "중식",
          "displayOrder": 7,
          "menuName": "배추김치/단호박죽"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
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
          "menuName": "참치김치찌개"
        },
        {
          "cornerName": "조식",
          "displayOrder": 4,
          "menuName": "닭양념오븐구이"
        },
        {
          "cornerName": "조식",
          "displayOrder": 5,
          "menuName": "감자양파매운볶음"
        },
        {
          "cornerName": "조식",
          "displayOrder": 6,
          "menuName": "열무된장무침"
        },
        {
          "cornerName": "조식",
          "displayOrder": 7,
          "menuName": "배추김치/요구르트"
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
          "menuName": "비요뜨"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
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
          "menuName": "닭개장"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "갈치무조림"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "깐풍만두"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "해물콩나물찜"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "얼갈이겉절이"
        },
        {
          "cornerName": "중식",
          "displayOrder": 7,
          "menuName": "배추김치/요구르트"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
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
          "menuName": "쇠고기무국"
        },
        {
          "cornerName": "조식",
          "displayOrder": 4,
          "menuName": "돼지가지볶음"
        },
        {
          "cornerName": "조식",
          "displayOrder": 5,
          "menuName": "두부조림"
        },
        {
          "cornerName": "조식",
          "displayOrder": 6,
          "menuName": "콩나물무침"
        },
        {
          "cornerName": "조식",
          "displayOrder": 7,
          "menuName": "배추김치/얼라이브쥬스"
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
          "menuName": "얼라이브쥬스"
        },
        {
          "cornerName": "조식",
          "displayOrder": 11,
          "menuName": "바나나"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "중식",
          "displayOrder": 1,
          "menuName": "도토리묵비빔밥&장볶이"
        },
        {
          "cornerName": "중식",
          "displayOrder": 2,
          "menuName": "유부국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "꿔바로우"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "오이양파무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "깍두기"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "배추김치/마시는 요거트"
        }
      ]
    }
  ]
}
```

</details>

</details>

<details>
<summary><strong>양식코너(안동)</strong> — HTTP 200, meals=5 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "국립경국대학교",
  "cafeteriaName": "양식코너(안동)",
  "sourceUrl": "https://www.gknu.ac.kr/?menu_idx=317",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "국립경국대학교",
    "cafeteriaName": "양식코너(안동)",
    "sourceUrl": "https://www.gknu.ac.kr/?menu_idx=317",
    "startDate": "2026-08-31",
    "endDate": "2026-09-04",
    "meals": [
      {
        "mealDate": "2026-08-31",
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
      },
      {
        "mealDate": "2026-09-01",
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
      },
      {
        "mealDate": "2026-09-02",
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
      },
      {
        "mealDate": "2026-09-03",
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
      },
      {
        "mealDate": "2026-09-04",
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
  "sourceUrl": "https://www.gknu.ac.kr/?menu_idx=317",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04",
  "meals": [
    {
      "mealDate": "2026-08-31",
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
    },
    {
      "mealDate": "2026-09-01",
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
    },
    {
      "mealDate": "2026-09-02",
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
    },
    {
      "mealDate": "2026-09-03",
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
    },
    {
      "mealDate": "2026-09-04",
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
<summary><strong>학생식당(예천)</strong> — HTTP 200, meals=14 (클릭해서 요청/응답 펼치기)</summary>

#### 요청

```json
{
  "schoolName": "국립경국대학교",
  "cafeteriaName": "학생식당(예천)",
  "sourceUrl": "https://www.gknu.ac.kr/?menu_idx=629",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04"
}
```

#### 응답 (`/api/v1/python/meals/crawl` 래핑)

```json
{
  "success": true,
  "data": {
    "schoolName": "국립경국대학교",
    "cafeteriaName": "학생식당(예천)",
    "sourceUrl": "https://www.gknu.ac.kr/?menu_idx=629",
    "startDate": "2026-08-31",
    "endDate": "2026-09-04",
    "meals": [
      {
        "mealDate": "2026-08-31",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "중식",
            "displayOrder": 1,
            "menuName": "뿌리채소영양밥&양념장"
          },
          {
            "cornerName": "중식",
            "displayOrder": 2,
            "menuName": "청포묵냉국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "오리불고기"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "계란찜"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "열무된장무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "배추김치"
          }
        ]
      },
      {
        "mealDate": "2026-08-31",
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
            "menuName": "돼지고기김치찌개"
          },
          {
            "cornerName": "석식",
            "displayOrder": 3,
            "menuName": "치킨까스&소스"
          },
          {
            "cornerName": "석식",
            "displayOrder": 4,
            "menuName": "비엔나감자조림"
          },
          {
            "cornerName": "석식",
            "displayOrder": 5,
            "menuName": "오이쪽파무침"
          },
          {
            "cornerName": "석식",
            "displayOrder": 6,
            "menuName": "배추김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
        "mealType": "BREAKFAST",
        "menus": [
          {
            "cornerName": "조식",
            "displayOrder": 1,
            "menuName": "쌀밥"
          },
          {
            "cornerName": "조식",
            "displayOrder": 2,
            "menuName": "수제비국"
          },
          {
            "cornerName": "조식",
            "displayOrder": 3,
            "menuName": "주꾸미볶음"
          },
          {
            "cornerName": "조식",
            "displayOrder": 4,
            "menuName": "두부찜*양념장"
          },
          {
            "cornerName": "조식",
            "displayOrder": 5,
            "menuName": "도시락김"
          },
          {
            "cornerName": "조식",
            "displayOrder": 6,
            "menuName": "배추김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
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
            "menuName": "꽃게된장국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "삼겹살고추장볶음"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "만두탕수"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "치커리사과생채"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "배추김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-01",
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
            "menuName": "짬뽕수제비국"
          },
          {
            "cornerName": "석식",
            "displayOrder": 3,
            "menuName": "고등어구이"
          },
          {
            "cornerName": "석식",
            "displayOrder": 4,
            "menuName": "미트볼피망조림"
          },
          {
            "cornerName": "석식",
            "displayOrder": 5,
            "menuName": "콩나물무침"
          },
          {
            "cornerName": "석식",
            "displayOrder": 6,
            "menuName": "배추김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
        "mealType": "BREAKFAST",
        "menus": [
          {
            "cornerName": "조식",
            "displayOrder": 1,
            "menuName": "[천원의아침밥]"
          },
          {
            "cornerName": "조식",
            "displayOrder": 2,
            "menuName": "쌀밥"
          },
          {
            "cornerName": "조식",
            "displayOrder": 3,
            "menuName": "순두부계란국"
          },
          {
            "cornerName": "조식",
            "displayOrder": 4,
            "menuName": "돈육불고기"
          },
          {
            "cornerName": "조식",
            "displayOrder": 5,
            "menuName": "쥐포채조림"
          },
          {
            "cornerName": "조식",
            "displayOrder": 6,
            "menuName": "숙주나물"
          },
          {
            "cornerName": "조식",
            "displayOrder": 7,
            "menuName": "배추김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
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
            "menuName": "참치김치찌개"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "안동찜닭"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "햄버섯볶음"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "부추겉절이"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "열무김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-02",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "석식",
            "displayOrder": 1,
            "menuName": "해물볶음우동"
          },
          {
            "cornerName": "석식",
            "displayOrder": 2,
            "menuName": "감자된장국"
          },
          {
            "cornerName": "석식",
            "displayOrder": 3,
            "menuName": "치킨너겟*머스타드"
          },
          {
            "cornerName": "석식",
            "displayOrder": 4,
            "menuName": "메밀전병*양념장"
          },
          {
            "cornerName": "석식",
            "displayOrder": 5,
            "menuName": "꼬들단무지"
          },
          {
            "cornerName": "석식",
            "displayOrder": 6,
            "menuName": "요플레"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
        "mealType": "BREAKFAST",
        "menus": [
          {
            "cornerName": "조식",
            "displayOrder": 1,
            "menuName": "쌀밥"
          },
          {
            "cornerName": "조식",
            "displayOrder": 2,
            "menuName": "만둣국"
          },
          {
            "cornerName": "조식",
            "displayOrder": 3,
            "menuName": "닭살고추장볶음"
          },
          {
            "cornerName": "조식",
            "displayOrder": 4,
            "menuName": "떡갈비&케찹"
          },
          {
            "cornerName": "조식",
            "displayOrder": 5,
            "menuName": "감자채볶음"
          },
          {
            "cornerName": "조식",
            "displayOrder": 6,
            "menuName": "배추김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
        "mealType": "LUNCH",
        "menus": [
          {
            "cornerName": "중식",
            "displayOrder": 1,
            "menuName": "중화덮밥"
          },
          {
            "cornerName": "중식",
            "displayOrder": 2,
            "menuName": "오이냉국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "돼지고기깐풍"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "메추리알조림"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "단무지"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "마카롱"
          }
        ]
      },
      {
        "mealDate": "2026-09-03",
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
            "menuName": "콩나물해장국"
          },
          {
            "cornerName": "석식",
            "displayOrder": 3,
            "menuName": "제육볶음"
          },
          {
            "cornerName": "석식",
            "displayOrder": 4,
            "menuName": "알감자조림"
          },
          {
            "cornerName": "석식",
            "displayOrder": 5,
            "menuName": "단배추들기름나물"
          },
          {
            "cornerName": "석식",
            "displayOrder": 6,
            "menuName": "총각김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
        "mealType": "BREAKFAST",
        "menus": [
          {
            "cornerName": "조식",
            "displayOrder": 1,
            "menuName": "[천원의아침밥]"
          },
          {
            "cornerName": "조식",
            "displayOrder": 2,
            "menuName": "쌀밥"
          },
          {
            "cornerName": "조식",
            "displayOrder": 3,
            "menuName": "오징어찌개"
          },
          {
            "cornerName": "조식",
            "displayOrder": 4,
            "menuName": "소불고기"
          },
          {
            "cornerName": "조식",
            "displayOrder": 5,
            "menuName": "맛살채소볶음"
          },
          {
            "cornerName": "조식",
            "displayOrder": 6,
            "menuName": "호박나물"
          },
          {
            "cornerName": "조식",
            "displayOrder": 7,
            "menuName": "배추김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
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
            "menuName": "들깨시래기된장국"
          },
          {
            "cornerName": "중식",
            "displayOrder": 3,
            "menuName": "닭갈비볶음"
          },
          {
            "cornerName": "중식",
            "displayOrder": 4,
            "menuName": "어묵볶음"
          },
          {
            "cornerName": "중식",
            "displayOrder": 5,
            "menuName": "도토리묵무침"
          },
          {
            "cornerName": "중식",
            "displayOrder": 6,
            "menuName": "배추김치"
          }
        ]
      },
      {
        "mealDate": "2026-09-04",
        "mealType": "DINNER",
        "menus": [
          {
            "cornerName": "석식",
            "displayOrder": 1,
            "menuName": "햄마요덮밥"
          },
          {
            "cornerName": "석식",
            "displayOrder": 2,
            "menuName": "우동국"
          },
          {
            "cornerName": "석식",
            "displayOrder": 3,
            "menuName": "감자크로켓&케찹"
          },
          {
            "cornerName": "석식",
            "displayOrder": 4,
            "menuName": "떡볶이"
          },
          {
            "cornerName": "석식",
            "displayOrder": 5,
            "menuName": "단무지"
          },
          {
            "cornerName": "석식",
            "displayOrder": 6,
            "menuName": "사과주스"
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
  "sourceUrl": "https://www.gknu.ac.kr/?menu_idx=629",
  "startDate": "2026-08-31",
  "endDate": "2026-09-04",
  "meals": [
    {
      "mealDate": "2026-08-31",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "중식",
          "displayOrder": 1,
          "menuName": "뿌리채소영양밥&양념장"
        },
        {
          "cornerName": "중식",
          "displayOrder": 2,
          "menuName": "청포묵냉국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "오리불고기"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "계란찜"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "열무된장무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "배추김치"
        }
      ]
    },
    {
      "mealDate": "2026-08-31",
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
          "menuName": "돼지고기김치찌개"
        },
        {
          "cornerName": "석식",
          "displayOrder": 3,
          "menuName": "치킨까스&소스"
        },
        {
          "cornerName": "석식",
          "displayOrder": 4,
          "menuName": "비엔나감자조림"
        },
        {
          "cornerName": "석식",
          "displayOrder": 5,
          "menuName": "오이쪽파무침"
        },
        {
          "cornerName": "석식",
          "displayOrder": 6,
          "menuName": "배추김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
      "mealType": "BREAKFAST",
      "menus": [
        {
          "cornerName": "조식",
          "displayOrder": 1,
          "menuName": "쌀밥"
        },
        {
          "cornerName": "조식",
          "displayOrder": 2,
          "menuName": "수제비국"
        },
        {
          "cornerName": "조식",
          "displayOrder": 3,
          "menuName": "주꾸미볶음"
        },
        {
          "cornerName": "조식",
          "displayOrder": 4,
          "menuName": "두부찜*양념장"
        },
        {
          "cornerName": "조식",
          "displayOrder": 5,
          "menuName": "도시락김"
        },
        {
          "cornerName": "조식",
          "displayOrder": 6,
          "menuName": "배추김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
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
          "menuName": "꽃게된장국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "삼겹살고추장볶음"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "만두탕수"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "치커리사과생채"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "배추김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-01",
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
          "menuName": "짬뽕수제비국"
        },
        {
          "cornerName": "석식",
          "displayOrder": 3,
          "menuName": "고등어구이"
        },
        {
          "cornerName": "석식",
          "displayOrder": 4,
          "menuName": "미트볼피망조림"
        },
        {
          "cornerName": "석식",
          "displayOrder": 5,
          "menuName": "콩나물무침"
        },
        {
          "cornerName": "석식",
          "displayOrder": 6,
          "menuName": "배추김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
      "mealType": "BREAKFAST",
      "menus": [
        {
          "cornerName": "조식",
          "displayOrder": 1,
          "menuName": "[천원의아침밥]"
        },
        {
          "cornerName": "조식",
          "displayOrder": 2,
          "menuName": "쌀밥"
        },
        {
          "cornerName": "조식",
          "displayOrder": 3,
          "menuName": "순두부계란국"
        },
        {
          "cornerName": "조식",
          "displayOrder": 4,
          "menuName": "돈육불고기"
        },
        {
          "cornerName": "조식",
          "displayOrder": 5,
          "menuName": "쥐포채조림"
        },
        {
          "cornerName": "조식",
          "displayOrder": 6,
          "menuName": "숙주나물"
        },
        {
          "cornerName": "조식",
          "displayOrder": 7,
          "menuName": "배추김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
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
          "menuName": "참치김치찌개"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "안동찜닭"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "햄버섯볶음"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "부추겉절이"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "열무김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-02",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "석식",
          "displayOrder": 1,
          "menuName": "해물볶음우동"
        },
        {
          "cornerName": "석식",
          "displayOrder": 2,
          "menuName": "감자된장국"
        },
        {
          "cornerName": "석식",
          "displayOrder": 3,
          "menuName": "치킨너겟*머스타드"
        },
        {
          "cornerName": "석식",
          "displayOrder": 4,
          "menuName": "메밀전병*양념장"
        },
        {
          "cornerName": "석식",
          "displayOrder": 5,
          "menuName": "꼬들단무지"
        },
        {
          "cornerName": "석식",
          "displayOrder": 6,
          "menuName": "요플레"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
      "mealType": "BREAKFAST",
      "menus": [
        {
          "cornerName": "조식",
          "displayOrder": 1,
          "menuName": "쌀밥"
        },
        {
          "cornerName": "조식",
          "displayOrder": 2,
          "menuName": "만둣국"
        },
        {
          "cornerName": "조식",
          "displayOrder": 3,
          "menuName": "닭살고추장볶음"
        },
        {
          "cornerName": "조식",
          "displayOrder": 4,
          "menuName": "떡갈비&케찹"
        },
        {
          "cornerName": "조식",
          "displayOrder": 5,
          "menuName": "감자채볶음"
        },
        {
          "cornerName": "조식",
          "displayOrder": 6,
          "menuName": "배추김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
      "mealType": "LUNCH",
      "menus": [
        {
          "cornerName": "중식",
          "displayOrder": 1,
          "menuName": "중화덮밥"
        },
        {
          "cornerName": "중식",
          "displayOrder": 2,
          "menuName": "오이냉국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "돼지고기깐풍"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "메추리알조림"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "단무지"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "마카롱"
        }
      ]
    },
    {
      "mealDate": "2026-09-03",
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
          "menuName": "콩나물해장국"
        },
        {
          "cornerName": "석식",
          "displayOrder": 3,
          "menuName": "제육볶음"
        },
        {
          "cornerName": "석식",
          "displayOrder": 4,
          "menuName": "알감자조림"
        },
        {
          "cornerName": "석식",
          "displayOrder": 5,
          "menuName": "단배추들기름나물"
        },
        {
          "cornerName": "석식",
          "displayOrder": 6,
          "menuName": "총각김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
      "mealType": "BREAKFAST",
      "menus": [
        {
          "cornerName": "조식",
          "displayOrder": 1,
          "menuName": "[천원의아침밥]"
        },
        {
          "cornerName": "조식",
          "displayOrder": 2,
          "menuName": "쌀밥"
        },
        {
          "cornerName": "조식",
          "displayOrder": 3,
          "menuName": "오징어찌개"
        },
        {
          "cornerName": "조식",
          "displayOrder": 4,
          "menuName": "소불고기"
        },
        {
          "cornerName": "조식",
          "displayOrder": 5,
          "menuName": "맛살채소볶음"
        },
        {
          "cornerName": "조식",
          "displayOrder": 6,
          "menuName": "호박나물"
        },
        {
          "cornerName": "조식",
          "displayOrder": 7,
          "menuName": "배추김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
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
          "menuName": "들깨시래기된장국"
        },
        {
          "cornerName": "중식",
          "displayOrder": 3,
          "menuName": "닭갈비볶음"
        },
        {
          "cornerName": "중식",
          "displayOrder": 4,
          "menuName": "어묵볶음"
        },
        {
          "cornerName": "중식",
          "displayOrder": 5,
          "menuName": "도토리묵무침"
        },
        {
          "cornerName": "중식",
          "displayOrder": 6,
          "menuName": "배추김치"
        }
      ]
    },
    {
      "mealDate": "2026-09-04",
      "mealType": "DINNER",
      "menus": [
        {
          "cornerName": "석식",
          "displayOrder": 1,
          "menuName": "햄마요덮밥"
        },
        {
          "cornerName": "석식",
          "displayOrder": 2,
          "menuName": "우동국"
        },
        {
          "cornerName": "석식",
          "displayOrder": 3,
          "menuName": "감자크로켓&케찹"
        },
        {
          "cornerName": "석식",
          "displayOrder": 4,
          "menuName": "떡볶이"
        },
        {
          "cornerName": "석식",
          "displayOrder": 5,
          "menuName": "단무지"
        },
        {
          "cornerName": "석식",
          "displayOrder": 6,
          "menuName": "사과주스"
        }
      ]
    }
  ]
}
```

</details>

</details>

---

