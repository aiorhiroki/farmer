from requests import Request, Session
import json


class PostClient:
    HEADERS = {'Content-Type': 'application/json'}

    def __init__(self, root_url):
        self.root_url = root_url
        self.obj_session = Session()

    def post(self, params: dict, route: str):
        obj_request = Request(
            "POST",
            self.root_url + route,
            data=json.dumps(params),
            headers=self.HEADERS
        )
        obj_prepped = self.obj_session.prepare_request(obj_request)
        res = self.obj_session.send(
            obj_prepped,
            verify=True,
            timeout=60
        )
        return res.json()

    def close_session(self):
        self.obj_session.close()
