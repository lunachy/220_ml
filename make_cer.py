# coding=utf-8
"""
Python一键创建个人CA证书

需要先安装openssl

如果浏览器出现`NET::ERR_CERT_INVALID`错误，需要把CA证书导入到浏览器中（双击CA证书即可），然后重启浏览器。
之后可能会出现`NET::ERR_CERT_AUTHORITY_INVALID`错误，不用理会，点击「高级」->「继续前往127.0.0.1（不安全）」即可访问。

示例：
python3 make_cer.py [argv]
argv:
  ca            创建CA证书
  server        创建server证书
  client        创建client证书
  all           创建以上所有证书
"""

import os
import sys
import socket

# 获取本地ip
# localIP = socket.gethostbyname(socket.gethostname())
localIP = "127.0.0.1"

# ca subj
country_name = "CN"
province_name = "GD"
locality_name = "GZ"
organization_name = "test"
organizational_unit_name = "test"
common_name = "root"
email = "admin@example.com"
password = ""

# server subj
s_country_name = "CN"
s_province_name = "GD"
s_locality_name = "GZ"
s_organization_name = "test"
s_organizational_unit_name = "test"
s_common_name = localIP
s_email = "admin@example.com"
s_password = "123456"

# client subj
c_country_name = "CN"
c_province_name = "GD"
c_locality_name = "GZ"
c_organization_name = "test"
c_organizational_unit_name = "test"
c_common_name = "user"
c_email = "admin@example.com"
c_password = "123456"

_subj_ca = "".join(
    (" -subj ", "/C=", country_name, "/ST=", province_name, "/L=",
     locality_name, "/O=", organization_name, "/OU=", organizational_unit_name,
     "/CN=", common_name, "/emailAddress=", email))
_subj_server = "".join(
    (" -subj ", "/C=", s_country_name, "/ST=", s_province_name, "/L=",
     s_locality_name, "/O=", s_organization_name, "/OU=",
     s_organizational_unit_name, "/CN=", s_common_name, "/emailAddress=",
     s_email))
_subj_client = "".join(
    (" -subj ", "/C=", c_country_name, "/ST=", c_province_name, "/L=",
     c_locality_name, "/O=", c_organization_name, "/OU=",
     c_organizational_unit_name, "/CN=", c_common_name, "/emailAddress=",
     c_email))

# 获取当前目录的绝对路径
curr_dir = os.path.dirname(os.path.realpath(__file__))

path_ca = os.path.join(curr_dir, "ca")
path_server = os.path.join(curr_dir, "server")
path_client = os.path.join(curr_dir, "client")

cakey_pem = os.path.join(path_ca, "ca-key.pem")
careq_csr = os.path.join(path_ca, "ca-req.csr")
cacert_pem = os.path.join(path_ca, "ca-cert.pem")
ca_p12 = os.path.join(path_ca, "ca.p12")

serverkey_pem = os.path.join(path_server, "server-key.pem")
severreq_csr = os.path.join(path_server, "server-req.csr")
servercert_pem = os.path.join(path_server, "server-cert.pem")
server_p12 = os.path.join(path_server, "server.p12")

clientkey_pem = os.path.join(path_client, "client-key.pem")
clientreq_csr = os.path.join(path_client, "client-req.csr")
clientcert_pem = os.path.join(path_client, "client-cert.pem")
client_p12 = os.path.join(path_client, "client.p12")

##########################################################################


def create_ca():
    """
    创建CA证书
    """

    if not os.path.exists(path_ca):
        os.mkdir(path_ca)
    # 创建根证书私钥
    os.system("openssl genrsa -out " + cakey_pem + " 1024 ")
    # 创建证书请求文件
    os.system(
        "openssl req -new -out " + careq_csr + " -key " + cakey_pem + _subj_ca)
    # 自签署证书
    os.system(
        "openssl x509 -req -in " + careq_csr + " -out " + cacert_pem +
        " -signkey " + cakey_pem + " -days 3650 " + "-passin pass:" + password)
    # 将证书导出成浏览器支持的.p12格式
    print("将证书导出成浏览器支持的.p12格式")
    os.system("openssl pkcs12 -export -clcerts -in " + cacert_pem + " -inkey "
              + cakey_pem + " -out " + ca_p12 + " -passout pass:" + password)


def create_server():
    """
    创建server证书
    """

    if not os.path.exists(path_server):
        os.mkdir(path_server)
    # 创建server私钥
    os.system("openssl genrsa -out " + serverkey_pem + " 1024 ")
    # 创建证书请求文件
    os.system("openssl req -new -out " + severreq_csr + " -key " +
              serverkey_pem + _subj_server)
    # 自签署证书
    os.system("openssl x509 -req -in " + severreq_csr + " -out " +
              servercert_pem + " -signkey " + serverkey_pem + " -CA " +
              cacert_pem + " -CAkey " + cakey_pem +
              " -CAcreateserial -days 3650 " + "-passin pass:" + s_password)
    # 将证书导出成浏览器支持的.p12格式
    os.system(
        "openssl pkcs12 -export -clcerts -in " + servercert_pem + " -inkey " +
        serverkey_pem + " -out " + server_p12 + " -passout pass:" + s_password)


def create_client():
    """
    创建client证书
    """

    if not os.path.exists(path_client):
        os.mkdir(path_client)
    # 创建私钥
    os.system("openssl genrsa -out " + clientkey_pem + " 1024 ")
    # 创建证书请求文件
    os.system("openssl req -new -out " + clientreq_csr + " -key " +
              clientkey_pem + _subj_client)
    # 自签署证书
    os.system("openssl x509 -req -in " + clientreq_csr + " -out " +
              clientcert_pem + " -signkey " + clientkey_pem + " -CA " +
              cacert_pem + " -CAkey " + cakey_pem +
              " -CAcreateserial -days 3650 " + "-passin pass:" + c_password)
    # 将证书导出成浏览器支持的.p12格式
    os.system(
        "openssl pkcs12 -export -clcerts -in " + clientcert_pem + " -inkey " +
        clientkey_pem + " -out " + client_p12 + " -passout pass:" + c_password)


def usage():
    print("  ca\t\t创建CA证书")
    print("  server\t创建server证书")
    print("  client\t创建client证书")
    print("  all\t\t创建以上所有证书")


def _argv():
    if len(sys.argv) > 1:
        value = sys.argv[1]
        if "ca" == value:
            create_ca()
        elif "server" == value:
            create_server()
        elif "client" == value:
            create_client()
        elif "all" == value:
            create_ca()
            create_server()
            create_client()
        else:
            usage()
        sys.exit()
    usage()


if __name__ == '__main__':
    _argv()
