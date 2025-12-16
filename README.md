🤖 RAG-Based Loose Matching Agent

Role-Differentiated LLM Architecture for Semantic Matching

이 레포지토리는
석사학위논문
「검색증강생성과 역할 분리형 LLM 기반 매칭 에이전트 설계」
의 실험 및 시스템 구현 코드입니다.

본 프로젝트는 비정형 자연어 메시지 간 Loose Matching 문제를 해결하기 위해
RAG(Retrieval-Augmented Generation) 와
역할 분리형 LLM 에이전트 구조를 결합한 매칭 시스템을 구현합니다.


---

🔍 Problem Statement

온라인 커뮤니티·플랫폼에서 이루어지는 매칭은 다음과 같은 한계를 가집니다.

사용자의 요구가 명시적으로 표현되지 않음

키워드 검색·임베딩 유사도 기반 방식은
👉 표현 다양성·맥락·역할 상보성을 놓침

단일 LLM 기반 매칭은
👉 복잡한 판단 기준을 일관되게 처리하기 어려움


이를 해결하기 위해 본 시스템은
“Loose Matching” 이라는 개념을 정의하고,
사람의 직관적 의미 판단을 역할 분리형 LLM 구조로 모사합니다.


---

🧠 Core Idea: Loose Matching

Loose Matching이란,

> 명시적 표현이 완전히 일치하지 않더라도
의도·맥락·역할 관점에서 의미적으로 연결 가능성이 있는 상대를 식별하는 매칭 방식


본 시스템은 다음 4가지 판단 기준을 중심으로 매칭을 수행합니다.

1. Matching Target & Hypernym
직접 대상 + 상위 개념 기반 의미 연결

2. 표현 명시성

암시적·정서적 표현까지 해석

3. 매칭 대상 유형(Type)
사람 / 장소·이벤트·서비스 / 물건

4. 역할 상보성 (Homogeneous vs Heterogeneous)



---

🏗 System Architecture

본 구현은 역할이 명확히 분리된 LLM 노드 기반 멀티 에이전트 구조로 설계되었습니다.

Main Nodes

Orchestrator	전체 흐름 제어, 노드 호출 및 재시도 관리
Query Reformer	의미 확장·역할 전환 기반 쿼리 재구성
Selector	다중 기준 기반 후보 선별
Evaluator	최종 매칭 판단 및 실패 관리


Auxiliary Nodes

Message Analyzer	사용자 메시지 의미 구조화
Web Search Module	고유명사·신조어 외부 지식 보강


---

🔄 Matching Flow

1. Input Message

2. Message Analysis / Web Search (optional)

3. Query Reformulation (2 parallel queries)

4. Vector Retrieval (FAISS)

5. Parallel Sub-Selectors
- PersonaMatch
- TypeMatch
- RoleMatch

6. Main Selector

7. Evaluator (max 2 iterations)


---

🧩 Selector Design

PersonaMatch: 성향·정서·페르소나 유사성

TypeMatch: 매칭 대상 유형 정합성

RoleMatch: 역할 상보성 (가장 중요한 기준)

Main Selector: 모든 판단을 종합해 단일 후보 선택


---

📦 Tech Stack

LLM
- OpenAI GPT-4o

Embedding
- text-embedding-3-small

Vector DB
- FAISS (cosine similarity)

External Tools
- Tavily Search API (Web Search Module)

---

🧪 Experiments

약 17,000건의 실제·합성 자연어 메시지 사용

Baseline(embedding-only) 실패 사례 80건 중
약 49%에서 의미적으로 설득력 있는 매칭 도출

역할 상보성이 중요한 시나리오에서 특히 성능 우수



---

📖 Reference

윤이지,
「검색증강생성과 역할 분리형 LLM 기반 매칭 에이전트 설계」,
경희대학교 빅데이터응용학과 석사학위논문, 2025.



---

⚠️ Notes

본 레포는 연구·실험 목적의 구현 코드입니다.

실제 서비스 적용 시:

비용 최적화 / 
LLM 호출 전략 / 
개인정보 보호 / 
사용자 선호 학습 로직 추가 고려 필요

