# check /eda/main.py

# 0. eda 마킹 및 결과물 기록

# 조졋다... 데이터셋 보니까.. STT한 데이터를 준거같아.. 스펠링이 많이 틀리는데..?

# 1. 1st eda - match with main.py

```
dataset_path :  /data/Tr2KB/train/train_data
label_path : /data/Tr2KB/train/train_label


dataset_path_list_dir top 10 :
['idx_188977.wav', 'idx_100906.wav', 'idx_187117.wav', 'idx_065846.wav', 'idx_188894.wav', 'idx_136961.wav', 'idx_119311.wav', 'idx_160989.wav', 'idx_118588.wav', 'idx_108600.wav']


len_dataset_path_list_dir : 200001

----------------------------------------------------------------------------------------------------
label_csv
          filename                                               text
0   idx_000000.wav   그러면 통화로 신청을 안 하고는 나중에 창구 가서 요청드려도 발급이 가능하다고 하...
1   idx_000001.wav   아 그렇군요.  저는 연 이자보다 수수료 무료가 드 중요한 것 같으니까 투천해주신...
2   idx_000002.wav   오 년 전쯤에 만든 통장이 거래가 없어서 거래 정지가 되어 있더라고요. 그래서 그...
3   idx_000003.wav              다름이 아니라 제가 국미눈앵 통장을 섀로 만들까 하고 연락드렸어요.
4   idx_000004.wav                          해지하는 게 나을까요? 안 하는 게 나을까요?
5   idx_000005.wav                        해지 먼저 도와드리겠습니다. 성함 말씀해 주세요.
6   idx_000006.wav                                 네 또 다른 문의사항 있으신가요?
7   idx_000007.wav                                           네 감사합니다.
8   idx_000008.wav                             네 상담사 김지현이었습니다. 감사합니다.
9   idx_000009.wav   전에 사용하던 통장은 입출금 통장인데 수수료 할인 안 해줘서 별론 것 같아서 새로...
10  idx_000010.wav                                통장 개설 목적은 어떻게 되실까요?
11  idx_000011.wav   다른 상품이랑 차이점이 있나요? 보니까 다른 통장은 자동이체 며익 건 하면 연 이...
12  idx_000012.wav   신규로 만드는 통장은 종류를 고를 수 있을까요? 제가 주로 월급 통장으로 사용학 ...
13  idx_000013.wav                                   근로 소득 생활비 사용입니다.
14  idx_000014.wav                                통신비 자도잇 자동이체 무료십니다.
15  idx_000015.wav                           네 그 상품은 수수료 무료 혜택이 없습니다.
16  idx_000016.wav                     아니면 지금 말슴드려야 종이 통장 발급 가능한 건가요?
17  idx_000017.wav                                            입력했습니다.
18  idx_000018.wav                        네 새로 개설할 통장 비밀번호 입력 부탁드립니다.
19  idx_000019.wav             인증 완료되셨구요. 기존에 가지고 계신 통장은 해지 도와도리겠습니다.
20  idx_000020.wav                 따로 추가금 같은 게 있을까요? 창고로 가서 제출하면 되나요?
21  idx_000021.wav               근처 국민은행 창구 가셔서 말씀해 주시면 당일 발급 가능하십니다.
22  idx_000022.wav   모바일 거래하려면 어플엣스 공인 인증서 인증하고 사용하면 되는 건가요? 국민 은행...
23  idx_000023.wav                           이제 신규 통장 개설 처리 도와드리겠습니다.
24  idx_000024.wav                       종이 통장도 발급 가능하게끔 처리 도와드리겠습니다.
25  idx_000025.wav                                       아니요 완료된 건가요?
26  idx_000026.wav                                           네 감사합니다.
27  idx_000027.wav                  그러면 어플에서 확인해보고 잘 모르겠으면 다시 연락드릴게요.
28  idx_000028.wav                                 그러면 새로 만드는 게 낫겠네요.
29  idx_000029.wav                                         네 등록되셨습니다.

------------------------------------------------------------------------------------------------------------
label_csv

columns : filename, text

```

# 2. 2nd eda

```
95% : --------------------------------------------------------------------------------
text         보통은 저처럼 카드를 발급한 게 확실하지만 발급한 카드를 찾지 못했을 경우에는 일...
text_len                                                  125
 보통은 저처럼 카드를 발급한 게 확실하지만 발급한 카드를 찾지 못했을 경우에는 일딸 일단 그 카드를 먼저 분실 처리를 진행한다는 말씀인 거죠? 그다음에는 재발급 서비스를 통해서 결국은 다시 발급을 받아야된다는 얘기 맞나요?

.99 :  --------------------------------------------------------------------------------
text         아 직접 가서 해도 된다고요? 아유 다행이네. 대체 이 놈의 공인인증서는 누가 만...
text_len                                                  156
 아 직접 가서 해도 된다고요? 아유 다행이네. 대체 이 놈의 공인인증서는 누가 만든 건지. 하여간 공무원들 책상 앞에서만 앉아 가지고 말이야. 아주 혼구녕을 내야 돼. 직접 가서 해도 공인인증서나 이런 게 필요한 건 아니죠? 내 주민등록증만 있으면 본인인 거 딱 알 거 아니야.

1. :  --------------------------------------------------------------------------------
text         네 안녕하세요. 대출 해피 대출 맞죠? 대출에 대해서 좀 여쭤보려고요. 제가 대출...
text_len                                                  297

 네 안녕하세요. 대출 해피 대출 맞죠? 대출에 대해서 좀 여쭤보려고요. 제가 대출을 하 어떻게 하는지를 하나도 몰라서요. 대충 대출 심사 기준이 있다는데 제가 신용점수도 칠백 오십으로 나쁘지 않은 점수를 가지고 있고 기존에 대출도 지금 하나도 없는 상태입니다. 그런데 요즘 경제가 어렵다 보니 목돈 마련이 좀 필요할 것 같아서요. 주위 사람들 다 빚져서라도 주식 투자하고 막 그러던데 저도 해야 그래야 하는 건지 걱정이 많네요. 그래도 아무 데서나 대출받을 수는 없다고 생각해서 해피 대출에 전화하게 되었습니다. 상담 가능할까요?

```

# 3. 3rd eda

- len_wav is calculated

```
                                           filename text  text_len  len_wav
38047   /data/Tr2KB/train/train_data/idx_038047.wav              1    22697
38036   /data/Tr2KB/train/train_data/idx_038036.wav              1    20051
176620  /data/Tr2KB/train/train_data/idx_176620.wav    다         1     9439
101348  /data/Tr2KB/train/train_data/idx_101348.wav              1    25919
53171   /data/Tr2KB/train/train_data/idx_053171.wav              1    21675
```

- check_data_format

```
1d array

[0.00097659 0.00091556 0.00085452 ... 0.00012207 0.00024415 0.00036622]
```

- sum wav_len

  > 19970411064
  > in seconds : 1_248_150 (if rate = 16_000)
  > in hours : 346 hours

- calculation duration : 1h 30min(너무 오래 걸리네.)

# 4. 4rd eda

```
text len 4 ----------------------------------------------------------------------------------------------------

['네 네.' '아 네.' '항공권 ' '그러면 ' '아 그럼' '오전모임' '아 네네' ' 아뇨.' '원리금 ' ' 있어요'
 '그럼 어' '그러지만' '가세요.' '있습니다' '아 상세' '이 월 ' '신청해 ' '입니다.' ' 거부해' '일억정도'
 '아아 네' '이자에 ' '네 뭐.' '일 회 ' '정석이건' '맞습니다' '예전에 ' ' 니다.' '고객님.' '핸드폰 '
 '이상없이' ' 제가 ' '됩니다.' '아니요.' '시티즌뱅' '빨리 발' '시 월 ' '지금은 ' '아하 네' ' 아니요'
 '되구요.' ' 네네.' '저는 김' '네 지금' '설정하고' ' 아 네' '네네. ' '어느 방' '습니다.' ' 그리고'
 '네습니다' '제가 증' ' 국세.' '하 이거']

len of data : 152

text len 3 ----------------------------------------------------------------------------------------------------
['니다.' '비자 ' ' 네.' '최고 ' '청년 ' '으 아' ' 넹.' '아니요' '네 네' '차영 ' '커피는' '네. '
 ' 없습' '카드가' '에이치']

len of data : 25

text len 2 ----------------------------------------------------------------------------------------------------
['한 ' ' 네' ' 음' '네.' '참 ' '나.' '이다' ' 아' '니다' '네 ']
len of data : 16

text len 1 ----------------------------------------------------------------------------------------------------
' ' '다'

len of data : 5
```

> number of na data : 199
> 제외하는게 맞을까?

```
roupby text_len nunique
          text
text_len
1            2
2           10
3           15
4           54
5          223
...        ...
289          2
290          1
291          2
294          2
297          1

[232 rows x 1 columns]
groupby text_len count
          text
text_len
1            5
2           16
3           25
4          152
5         1957
...        ...
289          8
290          1
291          2
294          2
297          1
```


# 5. MFCC max len eda
```
90%
torch.Size([80, 1301])

95%
torch.Size([80, 1635])

99%
torch.Size([80, 1743])


```