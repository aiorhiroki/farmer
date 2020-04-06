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
    user_list = [
        dict(
            name='hiroki',
            uid=1001,
            sudo='true',
        ),
        dict(
            name='atsushi',
            uid=1002,
            sudo='true',
            pub='pub key to be set'
        ),
        dict(
            name='yhamajima',
            uid=1003,
            sudo='false',
            pub='pub key to be set'
        ),
    ]

    return user_list


if __name__ == '__main__':
    main()