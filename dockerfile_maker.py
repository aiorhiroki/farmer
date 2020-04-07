import json
from jinja2 import Environment, FileSystemLoader


def main():

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template('Dockerfile.j2')

    users = get_user_list()

    arg_dic = dict(
        password='sigmoid',
        user_list=users
    )

    output = template.render(arg_dic)

    with open('Dockerfile', "w") as f:
        f.write(output)


def get_user_list():
    with open("user_setting.json", "r") as fr:
        user_list = json.load(fr)
    return user_list


if __name__ == '__main__':
    main()
