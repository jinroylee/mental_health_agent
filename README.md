Todo:

- 쓰레숄드 좀만 낮추자, 너무 안나온다
- runnablelambda

Historues:
- awaiting_feedback 만 보고 결정하던 로직 => feedback 을 분류하는 노드 추가 => 했음~
- MMR 로 query, instead of similarity search => 비슷한 서적이 많다보니
- Summarizing logic 5개까지만 window 에 저장
- custom parser for json inputs

Limitation:
- diagnoise 가 아쉬움
동공이 살짝 확장되고 식은땀이 나며 평소 110퍼센트의 심박수야. 나 왜이렇지? => 범용적 대답밖에 못함

- RAG 할떄 유저 메시지로 쿼리 해옴 => 가져온 문서를 바탕으로 히스토리 보고 거기에 답변 => 쿼리 할때에는 히스토리까지 쓰지를 않음 => 쿼리된 문서가 대화의 히스토리 안에서 유추해야 경우에 환각함
ex: anxiety 에 대해 배우고 싶어 => 답변 => "그거" 고치려면 어떻게 해야해 => 환각