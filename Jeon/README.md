1. zip하여 서버에 옮기기

```
$ zip -r Jeon.zip JeonRaDoSikDang

$ scp Jeon.zip kaic2023@101.101.209.54:~/

$ unzip Jeon.zip

```

2. setup.py
   > setup.py 에 의존성 있는 패키지 선언해야 docker생성시 반영됨.
   > glob2 패키지 설치시 오류가 발생하여, JH이 손수 코딩하여 대처함.
