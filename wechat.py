# coding=utf-8
import os
import itchat
from itchat.content import TEXT, PICTURE, RECORDING, ATTACHMENT, VIDEO
import time
from random import choice
import hashlib
import shutil
import requests

FLAG = 1
KEY = '8edce3ce905a4c1dbb965e6b35c3834d'


def calc_md5(img):
    with open(img, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def mv2unique_emoji(img):
    shutil.move(img, os.path.join(unique_emoji_dir, calc_md5(img)))


def get_signature():
    friends = itchat.get_friends(update=True)[0:]
    for i in friends:
        # 获取个性签名
        signature = i["Signature"]
        print signature


def get_response(msg):
    # 这里我们就像在“3. 实现最简单的与图灵机器人的交互”中做的一样
    # 构造了要发送给服务器的数据
    apiUrl = 'http://www.tuling123.com/openapi/api'
    data = {
        'key': KEY,
        'info': msg,
        'userid': 'wechat-robot',
    }
    try:
        r = requests.post(apiUrl, data=data).json()
        # 字典的get方法在字典没有'text'值的时候会返回None而不会抛出异常
        return r.get('text')
    # 为了防止服务器没有正常响应导致程序异常退出，这里用try-except捕获了异常
    # 如果服务器没能正常交互（返回非json或无法连接），那么就会进入下面的return
    except:
        # 将会返回一个None
        return


# 自动回复
# 封装好的装饰器，当接收到的消息是Text，即文字消息
@itchat.msg_register([TEXT])
def text_reply(msg):
    # 当消息不是由自己发出的时候
    if True:  # msg['FromUserName'] != myUserName or msg['FromUserName'] == msg['ToUserName']:
        print (u"[%s] From [%s] Message: %s\n" %
               (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg['CreateTime'])),
                msg['User']['NickName'], msg['Text']))

        # 为了保证在图灵Key出现问题的时候仍旧可以回复，这里设置一个默认回复
        default_reply = '~~~'
        # 如果图灵Key出现问题，那么reply将会是None
        reply = get_response(msg['Text'])
        print (u"\tReply Message: %s\n" % reply)
        # a or b的意思是，如果a有内容，那么返回a，否则返回b
        # 有内容一般就是指非空或者非None，你可以用`if a: print('True')`来测试
        time.sleep(2)
        print msg['FromUserName'], msg['ToUserName'], msg['User']['NickName']
        if msg['User']['NickName'] in [u'MRS静', u'Jessie']:
            itchat.send(reply, msg['FromUserName'])
            itchat.send(reply, msg['ToUserName'])
            # return reply or default_reply


# 处理群聊消息
# @itchat.msg_register([TEXT, PICTURE], isGroupChat=True)
def group_reply(msg):
    global FLAG
    # if msg['isAt']:
    # print msg['User']['NickName']  # 群名

    # myUserName: @ca355b0d4d8aea52fdb1a271761f2a70
    # badminton: @@530db8dd8559fc97e4682cba783ab7b86949bf30fbdbedc21d3e0b2918dfa4f6

    # ActualNickName    程
    # FromUserName @ca355b0d4d8aea52fdb1a271761f2a70
    # Type    Picture
    # ToUserName @@ 530db8dd8559fc97e4682cba783ab7b86949bf30fbdbedc21d3e0b2918dfa4f6
    # ActualUserName @ca355b0d4d8aea52fdb1a271761f2a70
    # IsAt    False
    # FileName    170817 - 162344.gif
    # Text < function download_fn at 0x7f584adc2ed8 >
    # CreateTime    1502958224
    # for i in msg:
    #     print i, msg[i]
    if msg['FromUserName'] == cr or msg['ToUserName'] == cr:
        if msg['Type'] == TEXT:
            print (u"[%s] From [Group: %s, Name: %s], Type: %s, Content: %s\n" %
                   (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg['CreateTime'])),
                    msg['User']['NickName'], msg['ActualNickName'], msg['Type'], msg['Content']))

        elif msg['Type'] == PICTURE:
            print (u"[%s] From [Group: %s, Name: %s], Type: %s, Content: %s\n" %
                   (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(msg['CreateTime'])),
                    msg['User']['NickName'], msg['ActualNickName'], msg['Type'], msg['FileName']))

            # 涉及版权问题，微信商店的表情不能下载
            if not msg['HasProductId']:
                img = os.path.join(download_emoji_dir, msg['FileName'])
                msg['Text'](img)
                shutil.move(img, os.path.join(unique_emoji_dir, calc_md5(img) + '.gif'))
            if FLAG:
                time.sleep(1)
                itchat.send('@img@{}'.format(os.path.join(unique_emoji_dir, choice(os.listdir(unique_emoji_dir)))), cr)
            FLAG = 1 - FLAG
            # return '@%s@%s' % ({'Picture': 'img', 'Video': 'vid'}.get(msg['Type'], 'fil'), msg['FileName'])


if __name__ == '__main__':
    # itchat.logout()
    download_emoji_dir = '/root/captcha/download_emoji/'
    unique_emoji_dir = '/root/captcha/unique_emoji/'
    # 如部分的linux系统，块字符的宽度为一个字符（正常应为两字符），故赋值为2
    itchat.auto_login(True, '/tmp/itchat1.pkl', enableCmdQR=2)

    # 获取自己的UserName
    myUserName = itchat.search_friends()['UserName']
    # name = itchat.search_friends(nickName='Jessie')['UserName']
    # print name
    # cr = itchat.search_chatrooms(u'飞起')[0]['UserName']
    # print ('myUserName: %s\nchatroom: %s' % (myUserName, cr))
    itchat.run()
