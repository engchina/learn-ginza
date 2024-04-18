import spacy
from spacy import displacy

spacy.require_cpu()
# spacy.prefer_gpu()
# spacy.require_gpu()

# print(f"===ja_ginza===")
# nlp = spacy.load('ja_ginza')
# doc = nlp('私は毎週水曜日にカフェで勉強します。その後、ジムに寄ってから帰ります。')
# for token in doc:
#     print(token)

# print(f"===ja_ginza_electra===")
# nlp = spacy.load('ja_ginza_electra')
# doc = nlp('夏の全国高等学校野球選手権大会に出場する')
# for token in doc:
#     print(token)
#
# print(f"===ja_ginza_electra 分割単位A===")
# nlp = spacy.load('ja_ginza_electra')
# ginza.set_split_mode(nlp, 'A')
# doc = nlp('夏の全国高等学校野球選手権大会に出場する')
# for token in doc:
#     print(token)
#
# print(f"===ja_ginza_electra 分割単位B===")
# nlp = spacy.load('ja_ginza_electra')
# ginza.set_split_mode(nlp, 'B')
# doc = nlp('夏の全国高等学校野球選手権大会に出場する')
# for token in doc:
#     print(token)
#
# print(f"===ja_ginza_electra 分割単位C===")
# nlp = spacy.load('ja_ginza_electra')
# ginza.set_split_mode(nlp, 'C')
# doc = nlp('夏の全国高等学校野球選手権大会に出場する')
# for token in doc:
#     print(token)

# GiNZAで固有表現抽出
# nlp = spacy.load('ja_ginza_electra')
# doc = nlp('小学生のサツキと5歳のメイの二人は、母の療養のために父と一緒に初夏の頃に3丁目に引っ越してくる。')
# for ent in doc.ents:
#     print(
#         ent.text + ',' +  # テキスト
#         ent.label_ + ',' +  # ラベル
#         str(ent.start_char) + ',' +  # 開始位置
#         str(ent.end_char)  # 終了位置
#     )

# GiNZAでルール追加
nlp = spacy.load('ja_ginza_electra')
nlp.add_pipe(factory_name='entity_ruler', config={"overwrite_ents": True}, last=True)
# Create an EntityRuler with overwrite entities enabled
patterns = [{'label': 'Person', 'pattern': '母'},
            {'label': 'Person', 'pattern': '父'}]
ruler = nlp.get_pipe('entity_ruler')
ruler.add_patterns(patterns)

doc = nlp('小学生のサツキと5歳のメイの二人は、母の療養のために父と一緒に初夏の頃に3丁目に引っ越してくる。')
# Print the entities found in the text
for ent in doc.ents:
    print(
        f"{ent.text},{ent.label_},{ent.label_},{ent.start_char},{ent.end_char}"  # Using f-string for better readability
    )
# for token in doc:
#     print(
#         token.text + ', ' +  # テキスト
#         token.tag_ + ', ' +  # SudachiPyの品詞タグ
#         token.pos_)  # Universal Dependenciesの品詞タグ

# 品詞の抽出（名詞のみ抽出する）
nlp = spacy.load('ja_ginza_electra')
doc = nlp('私は毎週水曜日にカフェで勉強します。その後、ジムに寄ってから帰ります。')

for token in doc:
    if '名詞' in token.tag_:
        print(token, token.tag_, token.pos_)

# レンマの抽出
nlp = spacy.load('ja_ginza_electra')
doc = nlp('私は毎週水曜日にカフェで勉強します。その後、ジムに寄ってから帰ります。')

# レンマと品詞の抽出
for sent in doc.sents:
    for token in sent:
        print(token.text, token.lemma_)

# 係り受け解析
nlp = spacy.load('ja_ginza_electra')
doc = nlp('私は毎週水曜日にカフェで勉強します。その後、ジムに寄ってから帰ります。')

for sent in doc.sents:
    for token in sent:
        print(token.text + ' ← ' + token.head.text + ', ' + token.dep_)

# 係り受け解析のグラフを表示する
# displacy.render(doc, style='dep', jupyter=True, options={'compact': True, 'distance': 90})
# displacy.serve(doc, style='dep', options={'compact': True, 'distance': 90})
# displacy.serve(doc, style='dep')

# 文境界解析
for sent in doc.sents:
    print(sent)
