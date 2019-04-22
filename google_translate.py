# coding=utf-8
import grequests
import logging
import json
import os
import time
from googletrans import Translator
from googletrans.utils import format_json
import sys
import re

translator = Translator(service_urls=['translate.google.cn'])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='log.txt')
logger = logging.getLogger()


def exception_handler(request, exception):
    logger.warning('exception when at %s :%s', request.url, exception)


def work(urls):
    reqs = (grequests.get(u, verify=True, allow_redirects=True, timeout=4) for u in urls)
    res = grequests.map(reqs, exception_handler=exception_handler, size=20)
    return res


def totaltranslate():
    # file2 = open('de2en_en.txt', mode='a', encoding='utf-8')

    with open('/data/chy/atomics/T1009/T1009.md', mode='r', encoding='utf-8') as f:
        urls = []
        num = 0
        for line in f:
            num += 1

            line = line.strip()
            token = translator.token_acquirer.do(line)
            token = '376431.207817'
            url = "https://translate.google.cn/translate_a/single?client=webapp&sl=en&tl=zh-CN&hl=zh-CN&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&pc=1&otf=1&ssel=3&tsel=0&kc=2&tk={0}&q={1}".format(
                token, line)
            urls.append(url)
            if len(urls) >= 50:
                res = work(urls)
                for r in res:
                    if hasattr(r, 'status_code'):
                        if r.status_code == 200:
                            try:
                                print(r.text)
                                # a = format_json(r.text)
                                # target = ''.join([d[0] if d[0] else '' for d in a[0]])
                                # source = ''.join([d[1] if d[1] else '' for d in a[0]])
                            except Exception as e:
                                logger.error('when format:%s', e)
                                logger.error('%s\n%s', r.text)
                                source = ''
                                target = ''
                        #     if len(source) != 0 and len(target) != 0:
                        #         file2.write(target + '\n')
                        #     else:
                        #         file2.write('\n')
                        # else:
                        #     file2.write('\n')
                urls = []
                logger.info('finish 50 sentence, now at %s', num)
    # file2.close()


def sentencetranslate(line):
    line = line.strip()
    text = translator.translate(line, src='de', dest='en').text
    return text


def completetranslate():
    file1 = open('de2en_en.txt', mode='r', encoding='utf-8')
    file2 = open('new_de2en_en.txt', mode='a', encoding='utf-8')
    i = 1
    with open('de.txt', mode='r', encoding='utf-8') as f:
        for line in f:
            t = file1.readline()
            if len(t) == 1:  # 'only \n'
                text = sentencetranslate(line)
                file2.write(text + '\n')
            else:
                file2.write(t)
            i += 1
            if i % 100 == 0:
                print(i)
    file1.close()
    file2.close()


def trans(file_path):
    with open(file_path) as f:
        all_text = ''
        for source in f:
            print(source)
            time.sleep(0.01)
            try:
                text = translator.translate(source, src='en', dest='zh-cn').text
                print(text)
                text = text.replace('＃', '#') + '\n'
                all_text += text
            except:
                return

    with open(file_path, 'w') as f, open(flag_txt, 'a') as f1:
        f.write(all_text)
        f1.write(file_path + '\n')


def fix(file_path):
    with open(file_path) as f:
        all_text = ''
        for source in f:
            all_text += source.replace('Atomic Test', '原子测试')

    with open(file_path, 'w') as f:
        f.write(all_text)


def trans_replace(word, line):
    text = translator.translate(word, src='en', dest='zh-cn').text
    return line.replace(word, text)


proxy = {
    'https': 'https://112.85.168.104:9999',
}


def trans_replace_html(file_path):
    all_text = ''
    with open(file_path) as f:
        lines = map(lambda x: x.strip(), f.readlines())
        line = ''.join(lines)
        # for line in f:
        trans_line = ''
        while line:
            # print(line)
            _i = line.find('>')
            trans_line += line[: _i + 1]
            line = line[_i + 1:]
            _j = line.find('<')
            if _j == -1:
                break
            word = line[:_j]
            line = line[_j:]
            if word.strip():
                translator = Translator(service_urls=['translate.google.cn'])
                text = translator.translate(word, src='en', dest='zh-cn').text
                print(word, text)
                trans_line += text
                # time.sleep(0.1)
        # print(trans_line)
        all_text = all_text + trans_line + '\n'

    with open(file_path, 'w') as f, open(flag_txt, 'a') as f1:
        f.write(all_text)
        f1.write(file_path + '\n')


if __name__ == "__main__":
    flag_txt = r'/tmp/flag.txt'
    with open(flag_txt) as f:
        processed_files = list(map(lambda line: line[:-1], f.readlines()))
    for root, dirs, files in os.walk(r'/data/chy/attack.mitre.org', topdown=False):
        for filename in files:
            file_path = os.path.join(root, filename)
            if r'\theme' not in root and file_path not in processed_files:
                print(file_path)
                trans_replace_html(file_path)

    sys.exit()
    translator = Translator(service_urls=['translate.google.cn'])
    for root, dirs, files in os.walk("/data/chy/atomics/", topdown=False):
        for filename in files:
            if filename.endswith('md'):
                file_path = os.path.join(root, filename)
                fix(file_path)
                # if file_path not in processed_files:
                #     trans(os.path.join(root, filename))
