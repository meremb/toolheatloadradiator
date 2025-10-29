from dash import Dash, html
from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.test import EnvironBuilder
from werkzeug.wrappers import Response

server = Flask(__name__)
app = Dash(__name__, server=server)
app.layout = html.Div([html.H1("Hello Dash on Vercel!")])

def handler(event, context):
    server.wsgi_app = ProxyFix(server.wsgi_app)
    builder = EnvironBuilder(
        path=event.get("path", "/"),
        method=event.get("httpMethod", "GET"),
        headers=event.get("headers", {}),
        query_string=event.get("queryStringParameters"),
        data=event.get("body", None),
    )
    env = builder.get_environ()
    resp = Response.from_app(server.wsgi_app, env)
    return {
        "statusCode": resp.status_code,
        "headers": dict(resp.headers),
        "body": resp.get_data(as_text=True),
    }

