from requests import Request, Session
import json


class MilkClient:
    HEADERS = {'Content-Type': 'application/json'}

    def __init__(self, config):
        self.api_url = config['MILK_API_URL']
        self.obj_session = Session()

    def post(self, params: dict):
        obj_request = Request(
            "POST",
            self.api_url,
            data=json.dumps(params),
            headers=self.HEADERS
        )
        obj_prepped = self.obj_session.prepare_request(obj_request)
        self.obj_session.send(
            obj_prepped,
            verify=True,
            timeout=60
        )

    def close_session(self):
        self.obj_session.close()
