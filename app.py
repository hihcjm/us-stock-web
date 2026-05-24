import logging
from api.app import app

# Render 환경에서 에러를 로그에 출력
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    app.run(debug=True)
