from app import server
from vercel_wsgi import handle

def handler(request, response):
    return handle(server, request, response)
