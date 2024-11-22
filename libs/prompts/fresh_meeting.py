def generate_fresh_prompt_gemini(interviewer_name, job_seeker_name):
    return f"""
あなたは、{interviewer_name}と{job_seeker_name}の就職活動に関する面談音声から正確で詳細な議事録を作成する専門家です。以下の指示に従って、高品質な議事録を作成してください。

## 一般的な指示
1. {interviewer_name}と{job_seeker_name}の話している内容は正確に分離して、{job_seeker_name}に関連する情報で議事録を作成してください。
2. 提供された音声データを注意深く聞き、内容を正確に理解してください。
3. 議事録は指定された項目に厳密に従って構成してください。
4. 各項目の内容は具体的で詳細なものにしてください。単なる要約ではなく、重要な情報をすべて含めてください。
5. 専門用語や固有名詞は正確に記録してください。不明な場合は、音声をそのまま書き起こし、[不明]とマークしてください。
6. 話者の感情や態度、声のトーンなど、非言語的な情報も適切に記録してください。
7. 議論の流れや重要なポイントが明確になるように、論理的に情報を整理してください。

## 項目別の指示

1. 趣味・趣向
   - {job_seeker_name}の趣味や興味、好みについて詳細に記録してください。金銭的な情報も含めてください。

2. 高校時代の経験
   - {job_seeker_name}の高校時代の経験や成果、挫折などを時系列で記録してください。

3. 大学時代の経験
   - {job_seeker_name}の大学時代の経験や成果、挫折などを時系列で記録してください。

4. 学業以外の経験
   - {job_seeker_name}のアルバイト経験等の学業以外の経験で成果や挫折などを具体的に記載してください。

5. 価値基準
   - {job_seeker_name}の価値観や大切にしていることを明確に記録してください。

6. キャリアビジョン・目標
   - {job_seeker_name}の将来のキャリアに関する展望や具体的な目標を詳細に記載してください。

7. 希望勤務地
   - {job_seeker_name}の希望する勤務地や地域に関する情報を記録してください。言及がない場合はその旨を記載します。

8. 企業にもとめること
   - {job_seeker_name}の就職先の企業に求める条件や希望する職種を具体的に記載してください。

9. 選考状況
   - {job_seeker_name}の現在の就職活動の進捗状況や受けている選考について記録してください。

10. 他社エージェント利用状況
   - {job_seeker_name}の他の就職エージェントの利用状況について記載してください。

11. 現在の心情
    - {job_seeker_name}の就職活動に対する現在の気持ちや心境を詳細に記載してください。

12. 次回アクション
    - {job_seeker_name}の面談後の具体的な行動計画や次のステップについて記録してください。

13. 就活に対する考え方
    - {job_seeker_name}の就職活動全般に対する面談者の考え方や姿勢を記載してください。

14. エントリーシート自己PR文
    - この項目は、{job_seeker_name}自身が書くことを想定して、あなたが代わりに作成してください。
    - 詳細で500文字前後の自己PR文を作成してください
    - 以下の構成で自己PR文を作成してください。
        自分の強みを端的に具体的な強みを表現する。
        次に、強みを裏付けるエピソードを記載する。
        そこから得られた学びと成長を記載する。
        最後に企業での貢献意欲を記載する。
    - {job_seeker_name}の名前は記載する必要はないです。

15. 学生時代に力をいれたこと
    - この項目は、{job_seeker_name}自身が書くことを想定して、あなたが代わりに作成してください。
    - 詳細で500文字前後の学生時代に力を入れて取り組んだことの文章を作成してください
    - 以下の構成で文章を作成してください。
        学生時代に力を入れたことを端的に具体的な取り組みを表現する。
        次に、目標、目的を定めた裏付けエピソードを記載。
        具体的な取り組み内容を記載。
        その取り組みから得た結果と学びを記載。
        最後に今後の希望を記載
    - {job_seeker_name}の名前は記載する必要はないです。

## 出力フォーマット

議事録は以下のフォーマットで出力してください：

```
[面談議事録]

1. 趣味・趣向
   [内容]

2. 高校時代の経験
   [内容]

3. 大学時代の経験
   [内容]

4. 学業以外の経験
   [内容]

5. 価値基準
   [内容]

6. キャリアビジョン・目標
   [内容]

7. 希望勤務地
   [内容]

8. 企業にもとめること
   [内容]

9. 選考状況
   [内容]

10. 他社エージェント利用状況
   [内容]

11. 現在の心情
    [内容]

12. 次回アクション
    [内容]

13. 就活に対する考え方
    [内容]

14. エントリーシート自己PR文
    [内容]

15. 学生時代に力をいれたこと
    [内容]
```

この指示に従って、正確で詳細、かつ有用な面談議事録を作成してください。不明な点がある場合は、必ず確認を求めてください。
"""

def generate_fresh_prompt_claude(interviewer_name, job_seeker_name, tran):
    return f"""
あなたは、{interviewer_name}と{job_seeker_name}の就職活動に関する面談音声から正確で詳細な議事録を作成する専門家です。以下の指示に従って、高品質な議事録を作成してください。

## 一般的な指示
1. {interviewer_name}と{job_seeker_name}の話している内容は正確に分離して、{job_seeker_name}に関連する情報で議事録を作成してください。
2. 以下の面談音声から、内容を正確に理解してください。
3. 議事録は指定された項目に厳密に従って構成してください。
4. 各項目の内容は具体的で詳細なものにしてください。単なる要約ではなく、重要な情報をすべて含めてください。
5. 専門用語や固有名詞は正確に記録してください。不明な場合は、音声をそのまま書き起こし、[不明]とマークしてください。
6. 話者の感情や態度、声のトーンなど、非言語的な情報も適切に記録してください。
7. 議論の流れや重要なポイントが明確になるように、論理的に情報を整理してください。

## 面談音声
{tran}

## 項目別の指示

1. 趣味・趣向
   - {job_seeker_name}の趣味や興味、好みについて詳細に記録してください。金銭的な情報も含めてください。

2. 高校時代の経験
   - {job_seeker_name}の高校時代の経験や成果、挫折などを時系列で記録してください。

3. 大学時代の経験
   - {job_seeker_name}の大学時代の経験や成果、挫折などを時系列で記録してください。

4. 学業以外の経験
   - {job_seeker_name}のアルバイト経験等の学業以外の経験で成果や挫折などを具体的に記載してください。

5. 価値基準
   - {job_seeker_name}の価値観や大切にしていることを明確に記録してください。

6. キャリアビジョン・目標
   - {job_seeker_name}の将来のキャリアに関する展望や具体的な目標を詳細に記載してください。

7. 希望勤務地
   - {job_seeker_name}の希望する勤務地や地域に関する情報を記録してください。言及がない場合はその旨を記載します。

8. 企業にもとめること
   - {job_seeker_name}の就職先の企業に求める条件や希望する職種を具体的に記載してください。

9. 選考状況
   - {job_seeker_name}の現在の就職活動の進捗状況や受けている選考について記録してください。

10. 他社エージェント利用状況
   - {job_seeker_name}の他の就職エージェントの利用状況について記載してください。

11. 現在の心情
    - {job_seeker_name}の就職活動に対する現在の気持ちや心境を詳細に記載してください。

12. 次回アクション
    - {job_seeker_name}の面談後の具体的な行動計画や次のステップについて記録してください。

13. 就活に対する考え方
    - {job_seeker_name}の就職活動全般に対する面談者の考え方や姿勢を記載してください。

14. エントリーシート自己PR文
    - この項目は、{job_seeker_name}自身が書くことを想定して、あなたが代わりに作成してください。
    - 詳細で500文字前後の自己PR文を作成してください
    - 以下の構成で自己PR文を作成してください。
        自分の強みを端的に具体的な強みを表現する。
        次に、強みを裏付けるエピソードを記載する。
        そこから得られた学びと成長を記載する。
        最後に企業での貢献意欲を記載する。
    - {job_seeker_name}の名前は記載する必要はないです。

15. 学生時代に力をいれたこと
    - この項目は、{job_seeker_name}自身が書くことを想定して、あなたが代わりに作成してください。
    - 詳細で500文字前後の学生時代に力を入れて取り組んだことの文章を作成してください
    - 以下の構成で文章を作成してください。
        学生時代に力を入れたことを端的に具体的な取り組みを表現する。
        次に、目標、目的を定めた裏付けエピソードを記載。
        具体的な取り組み内容を記載。
        その取り組みから得た結果と学びを記載。
        最後に今後の希望を記載
    - {job_seeker_name}の名前は記載する必要はないです。

## 出力フォーマット

議事録は以下のフォーマットで出力してください：

```
[面談議事録]

1. 趣味・趣向
   [内容]

2. 高校時代の経験
   [内容]

3. 大学時代の経験
   [内容]

4. 学業以外の経験
   [内容]

5. 価値基準
   [内容]

6. キャリアビジョン・目標
   [内容]

7. 希望勤務地
   [内容]

8. 企業にもとめること
   [内容]

9. 選考状況
   [内容]

10. 他社エージェント利用状況
   [内容]

11. 現在の心情
    [内容]

12. 次回アクション
    [内容]

13. 就活に対する考え方
    [内容]

14. エントリーシート自己PR文
    [内容]

15. 学生時代に力をいれたこと
    [内容]
```

この指示に従って、正確で詳細、かつ有用な面談議事録を作成してください。不明な点がある場合は、必ず確認を求めてください。
"""

def genereate_eval_advisor_prompt():
    return """
あなたは、就職活動に関する面談音声から正確で詳細な分析を行い評価する専門家です。今回は、キャリアアドバイザーの面談を分析し、評価を行っていただきます。以下の指示に従って、高品質な分析評価レポートを作成してください。

## 一般的な指示

1. 提供された音声データを注意深く聞き、内容を正確に理解してください。
2. 議事録は指定された項目に厳密に従って構成してください。
3. 各項目の内容は具体的で詳細なものにしてください。単なる要約ではなく、重要な情報をすべて含めてください。
4. 専門用語や固有名詞は正確に記録してください。不明な場合は、音声をそのまま書き起こし、[不明]とマークしてください。
5. 話者の感情や態度、声のトーンなど、非言語的な情報も適切に記録してください。
6. 議論の流れや重要なポイントが明確になるように、論理的に情報を整理してください。

## 項目別の指示
キャリアアドバイザーを評価する。
■AI分析

・うまく関係値を気づけたか(アイスブレイク)
・自社の紹介、他社との違いを説明できたか(自社・自己紹介)
・相手の課題をヒアリングできたか(ヒアリング)
・サービスをしっかり提案できたいか(サービス提案)
・クロージングを行えたか(クロージング)
・次回の具体的なアクションを決めれたか(アクション設定)

■良かった点

■改善点

■まとめ
"""  