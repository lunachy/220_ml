# coding: utf-8
import requests
from time import sleep, time
import shutil
import os
import sys
import json
from multiprocessing import cpu_count, Pool

pic_path = '/root/captcha/rest_code_pic'
code_pic_path = '/root/captcha/code_test_pic'

CPU_COUNT = cpu_count()


def main(file_name):
    '''
            main() 参数介绍
            api_username    （API账号）             --必须提供
            api_password    （API账号密码）         --必须提供
            file_name       （需要打码的图片路径）   --必须提供
            api_post_url    （API接口地址）         --必须提供
            yzm_min         （验证码最小值）        --可空提供
            yzm_max         （验证码最大值）        --可空提供
            yzm_type        （验证码类型）          --可空提供
            tools_token     （工具或软件token）     --可空提供
    '''
    # api_username =
    # api_password = 
    # file_name = 'c:/temp/lianzhong_vcode.png'
    # api_post_url = "http://v1-http-api.jsdama.com/api.php?mod=php&act=upload"
    # yzm_min = '1'
    # yzm_max = '8'
    # yzm_type = '1303'
    # tools_token = api_username

    # proxies = {'http': 'http://127.0.0.1:8888'}
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
        'Accept-Encoding': 'gzip, deflate',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:53.0) Gecko/20100101 Firefox/53.0',
        # 'Content-Type': 'multipart/form-data; boundary=---------------------------227973204131376',
        'Connection': 'keep-alive',
        'Host': 'v1-http-api.jsdama.com',
        'Upgrade-Insecure-Requests': '1'
    }

    files = {
        'upload': (file_name, open(file_name, 'rb'), 'image/png')
    }
    data = {
        'user_name': 'abcd51685168',
        'user_pw': 'Abcd5168*',
        'yzm_minlen': '4',
        'yzm_maxlen': '4',
        'yzmtype_mark': '1001',
        'zztool_token': 'abcd51685168'
    }

    s = requests.session()
    r = s.post('http://v1-http-api.jsdama.com/api.php?mod=php&act=upload',
               headers=headers, data=data, files=files, verify=False)

    # {"result": true, "data": {"id": 8353571068, "val": "TYSK"}}
    # {"data": "", "result": false}
    result = json.loads(r.text)
    if result['result']:
        code = result['data']['val']
        shutil.move(file_name, os.path.join(code_pic_path, code + '.jpg'))
        print('rename {} to {} success!'.format(file_name, code))
    else:
        print('{} failed'.format(filename))


if __name__ == '__main__':
    os.chdir(pic_path)
    start = time()
    # for filename in os.listdir(pic_path):
    #     main(filename)

    pool = Pool(processes=CPU_COUNT, maxtasksperchild=400)
    for filename in os.listdir(pic_path):
        pool.apply_async(main, (filename,))
    pool.close()
    pool.join()
    end = time()
    print("cost all time: %s seconds." % (end - start))
