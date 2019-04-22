#!/usr/bin/python
# coding=utf-8

from selenium import webdriver
from time import sleep
import re
import sys
import os


def read_emails(filepath):
    with open(filepath) as f:
        return map(lambda x: x.split(), f.readlines())


def expand_shadow_element(driver, element):
    return driver.execute_script('return arguments[0].shadowRoot', element)


def acvive_vt_url(driver, user, password):
    driver.get('https://mail.163.com/')
    sleep(2)

    # login
    driver.switch_to.frame(driver.find_element_by_xpath("//iframe[starts-with(@id, 'x-URS-iframe')]"))
    # driver.switch_to.frame(2)
    driver.find_element_by_name("email").clear()
    driver.find_element_by_name("email").send_keys(user)
    driver.find_element_by_name("password").send_keys(password)
    driver.find_element_by_id("dologin").click()
    sleep(2)
    driver.find_element_by_link_text("继续登录").click()
    sleep(2)
    driver.switch_to.default_content()

    # open receiver
    driver.find_elements_by_class_name("oz0")[0].click()  # 点击收信
    sleep(2)
    driver.find_elements_by_class_name("dP0")[0].click()
    driver.switch_to.frame(driver.find_element_by_class_name("oD0"))
    href = driver.find_element_by_xpath("//pre").text
    vt_url = re.findall(r'https://www.virustotal.com/gui/account-activation/\w+', href)[0]
    driver.switch_to.default_content()
    driver.get(vt_url)
    sleep(2)


def get_vt_key(driver, vt_url, user, password):
    driver.get(vt_url)
    sleep(2)

    # login vt
    print('open signin url')
    root1 = driver.find_element_by_tag_name('vt-virustotal-app')
    shadow1 = expand_shadow_element(driver, root1)
    root2 = shadow1.find_element_by_css_selector('sign-in-view')
    shadow2 = expand_shadow_element(driver, root2)
    root_email = shadow2.find_elements_by_id('email')[0]
    shadow_email = expand_shadow_element(driver, root_email)
    shadow_email.find_element_by_id('input').send_keys(user)
    root_password = shadow2.find_elements_by_id('password')[0]
    shadow_password = expand_shadow_element(driver, root_password)
    shadow_password.find_element_by_id('input').send_keys(password)
    sleep(1)
    shadow2.find_element_by_css_selector('vt-ui-button').click()
    print('vt-ui-button')
    sleep(20)

    # get api key
    print('open apikey url: ' + user)
    driver.get('https://www.virustotal.com/gui/user/{}/apikey'.format(user))
    print('virustotal apikey url: ' + user)
    sleep(3)

    root1 = driver.find_element_by_tag_name('vt-virustotal-app')
    shadow1 = expand_shadow_element(driver, root1)
    root2 = shadow1.find_element_by_css_selector('apikey-view')
    shadow2 = expand_shadow_element(driver, root2)
    vt_key = shadow2.find_element_by_css_selector('div[slot][style]').find_element_by_css_selector('div').text
    print(user + ' vt_key: ' + vt_key)
    return vt_key


if __name__ == '__main__':
    active_mail = 'active_mail.txt'
    out_file = 'mail163_a.txt'
    with open(active_mail) as f:
        users = list(map(lambda x: x.strip(), f.readlines()))
    with open(out_file) as f:
        users_out = list(map(lambda x: x.split()[0].split('@')[0], f.readlines()))
    vt_login_url = 'https://www.virustotal.com/gui/sign-in'
    emails = list(read_emails('mail163.txt'))

    with open(active_mail, 'a') as f:
        for user, password in emails:
            user = user.split('@')[0]
            if user in users:
                continue
            print('processing: ' + user)
            driver = webdriver.Chrome()
            acvive_vt_url(driver, user, password)
            driver.close()
            f.write(user + '\n')
    with open(out_file, 'a') as f:
        for _user, _passwd in emails:
            user = _user.split('@')[0]
            password = user + user
            if user not in users_out:
                driver = webdriver.Chrome()
                vt_key = get_vt_key(driver, vt_login_url, user, password)
                driver.close()
                if vt_key:
                    f.write('  '.join([_user, _passwd, vt_key]) + '\n')
