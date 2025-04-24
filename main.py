from bottle import Bottle
import tempfile

def safe_del(self):
    try:
        self.close()
    except FileNotFoundError:
        pass

tempfile._TemporaryFileCloser.__del__ = safe_del

from FeatureCloud.app.api.http_ctrl import api_server
from FeatureCloud.app.api.http_web import web_server

from FeatureCloud.app.engine.app import app

import states

server = Bottle()


if __name__ == '__main__':
    app.register()
    server.mount('/api', api_server)
    server.mount('/web', web_server)
    server.run(host='localhost', port=5000)
