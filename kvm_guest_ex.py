#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
from time import sleep, time
import argparse
import xmlrpclib
import threading
import sys
import Queue
from collections import namedtuple
import logging
import logging.handlers
from xml.etree.ElementTree import register_namespace, parse
import ConfigParser
import psutil

conf = ConfigParser.ConfigParser()

LOG_ROOT = "/data/log/"
log = logging.getLogger(__file__)
formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

fh = logging.handlers.WatchedFileHandler(os.path.join(LOG_ROOT, 'kvm_guest_ex.log'))
fh.setFormatter(formatter)
log.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
log.addHandler(ch)

log.setLevel(logging.INFO)

XML_PATH = "/data/image/"
BACKUP_PATH = "/data/image/backup/"
QEMU_PATH = "/etc/libvirt/qemu/"
BACKUP_XML_PATH = XML_PATH + "template.xml"

INIT_VM = namedtuple('INIT_VM', ['type', 'mac', 'num', 'start_ip', 'memory', 'uuid'])
DETAIL_VM = namedtuple('DETAIL_VM', ['domain', 'ip', 'mac', 'memory', 'uuid', 'backup_qcow2'])
TAIL = "> /dev/null"
STATUS_INIT = 0x0001
DEFAULT_CPU_COUNT = 24
# Vm may start abnormally sometimes, it should try restart vm MAX_COUNT times.
MAX_COUNT = 3
VMS = []
DOMAIN_IP_MAC = []


def exec_log(cmd):
    os.system(cmd)
    log.info(cmd)


def generate_kvm_conf():
    kvm_path = "/root/.cuckoo/conf/kvm.conf"
    platform = "windows"
    conf.add_section("kvm")
    machines = []
    vm_types = set()
    for vm in VMS:
        label = vm.domain
        primary_os = label.split('_')[0]
        if primary_os == "winxpl":
            primary_os = "winxp"
        vm_types.add(primary_os)
        machines.append(label)
        conf.add_section(label)
        conf.set(label, "label", label)
        conf.set(label, "platform", platform)
        conf.set(label, "ip", vm.ip)
        conf.set(label, "tags", primary_os)
    conf.set("kvm", "machines", ",".join(machines))
    # conf.set("kvm", "vm_types", ",".join(vm_types))
    conf.set("kvm", "interface", "br0")
    with open(kvm_path, "w") as f:
        conf.write(f)


class VmOperation(object):
    def __init__(self):
        self.prefix_ip = "10.14.24."
        # num -- number of vms
        # start_ip -- start_ip of domain index and ip
        self.init_vms = [
            # INIT_VM("win2k3_sp2s_",      "52:54:00:33:10:",  2,  20,  "524288", "fca2a5fd-f42c-4a62-53f0-9253bde310"),
            # INIT_VM("winxp_sp2s_",       "52:54:00:33:11:",  2,  26,  "524288", "fca2a5fd-f42c-4a62-53f0-9253bde311"),
            INIT_VM("winxp_sp3_", "52:54:00:33:a4:", 20, 50, "524288", "fca2a5fd-f42c-4a62-53f0-9253bde312"),
            # INIT_VM("win7_sp1_32s_",     "52:54:00:33:13:",  2,  36, "1048576", "fca2a5fd-f42c-4a62-53f0-9253bde313"),
            #
            # INIT_VM("winxp_sp3_10l_",    "52:54:00:33:19:", 10, 115,  "524288", "fca2a5fd-f42c-4a62-53f0-9253bde319"),
            # INIT_VM("winxp_sp2l_",       "52:54:00:33:16:",  2,  70,  "524288", "fca2a5fd-f42c-4a62-53f0-9253bde316"),
            # INIT_VM("winxp_sp3_03l_",    "52:54:00:33:17:",  2,  85,  "524288", "fca2a5fd-f42c-4a62-53f0-9253bde317"),
            # INIT_VM("winxp_sp3_07l_",    "52:54:00:33:18:",  2, 100,  "524288", "fca2a5fd-f42c-4a62-53f0-9253bde318"),
            # INIT_VM("win7_32l_",         "52:54:00:33:1a:",  2, 145, "1048576", "fca2a5fd-f42c-4a62-53f0-9253bde31a"),
            # INIT_VM("win7_sp1_32l_",     "52:54:00:33:1b:", 10, 155, "1048576", "fca2a5fd-f42c-4a62-53f0-9253bde31b"),
            # INIT_VM("win2k3_sp2l_",      "52:54:00:33:14:",  2,  50,  "524288", "fca2a5fd-f42c-4a62-53f0-9253bde314"),
            # INIT_VM("winxpl_",           "52:54:00:33:15:",  2,  60,  "524288", "fca2a5fd-f42c-4a62-53f0-9253bde315"),
            #
            # INIT_VM("win7_sp1_64l_",     "52:54:00:33:1c:",  2, 185, "1572864", "fca2a5fd-f42c-4a62-53f0-9253bde31c")
        ]

    # generate rem domain-ip-mac
    def gen_rem_ip_mac(self):
        for vm in self.init_vms:
            title = "\nrem {0} ip-mac".format(vm.type.replace('_', '-'))
            DOMAIN_IP_MAC.append(title)
            for i in range(vm.num):
                domain = vm.type + str(i)
                mac = vm.mac.replace(':', '-') + str(i + 50)
                ip = self.prefix_ip + str(i + vm.start_ip)
                DOMAIN_IP_MAC.append(" ".join(["rem", domain, mac, ip]))

    def get_vm_ip_by_mac(self):
        for vm in self.init_vms:
            backup_qcow2 = vm.type + "backup.qcow2"
            for i in range(vm.num):
                domain = vm.type + str(i)
                mac = vm.mac + str(i + 50)
                ip = self.prefix_ip + str(i + vm.start_ip)
                uuid = vm.uuid + str(i + 10)
                VMS.append(DETAIL_VM(domain, ip, mac, vm.memory, uuid, backup_qcow2))

    def define(self):
        cpu_count = psutil.cpu_count()
        if cpu_count != DEFAULT_CPU_COUNT:
            cmd = "sed -i 's/1-{0}/1-{1}/g' `grep '1-{0}' -l *xml`".format(DEFAULT_CPU_COUNT - 1, cpu_count - 1)
            os.system(cmd)
        for vm in VMS:
            exec_log("virsh define {}.xml {}".format(vm.domain, TAIL))
            exec_log("virsh snapshot-create {0} {0}_snapshot.xml --redefine --current".format(vm.domain, TAIL))

    def destroy(self):
        for vm in VMS:
            exec_log("virsh destroy {} {}".format(vm.domain, TAIL))

    # rude method
    def undefine(self):
        self.destroy()
        sleep(2)
        for vm in VMS:
            xml_path = QEMU_PATH + vm.domain + ".xml"
            exec_log("rm {} {}".format(xml_path, TAIL))
        exec_log("service libvirt-bin restart")

    def list_snapshot(self, domain=None):
        if domain:
            ret = os.popen("virsh snapshot-list {} | grep running".format(domain))
            log.info(domain, ret.read())
        else:
            for vm in VMS:
                ret = os.popen("virsh snapshot-list {} | grep running".format(vm.domain))
                log.info("%s, %s" %(vm.domain, ret.read()))

    def rm_vm(self):
        for vm in VMS:
            exec_log("rm {}* {}".format(vm.domain, TAIL))

    def test_vm(self):
        revert_timeout = 10
        time_step = 1
        port = 8000
        for vm in VMS:
            exec_log("virsh snapshot-revert {} --current {}".format(vm.domain, TAIL))
            ret = wait_init(vm.ip, port, revert_timeout, time_step)
            if ret:
                log.info("{} revert successfully.".format(vm.domain))
            else:
                log.warning("{} revert failed.".format(vm.domain))
            exec_log("virsh destroy {} {}".format(vm.domain, TAIL))
            sleep(1)


def create_xml(domain, uuid, memory, mac):
    register_namespace("qemu", "http://libvirt.org/schemas/domain/qemu/1.0")
    tree = parse(BACKUP_XML_PATH)
    root = tree.getroot()
    root.find("name").text = domain
    root.find("uuid").text = uuid
    root.find("memory").text = memory
    root.find("currentMemory").text = memory
    root.find("devices").find("disk").find("source").set("file", XML_PATH + domain + ".qcow2")
    root.find("devices").find("interface").find("mac").set("address", mac)
    tree.write(domain + ".xml")


def wait_init(ip, port, timeout, time_step):
    url = "http://{0}:{1}".format(ip, port)
    server = xmlrpclib.ServerProxy(url)
    cost_time = 0

    while True:
        if cost_time > timeout:
            return False
        try:
            status = server.get_status()
            if status == STATUS_INIT:
                return True
        # error: [Errno 113] No route to host
        except Exception as e:
            # if e.errno == 113:
            sleep(time_step)
            cost_time += time_step
            continue


def start_vm(domain, ip, port=8000):
    start_timeout = 600
    time_step = 5
    for i in range(MAX_COUNT):
        exec_log("virsh start {} {}".format(domain, TAIL))
        start_ret = wait_init(ip, port, start_timeout, time_step)
        if start_ret:
            log.info("time:%s, %s start successfully" % (i + 1, domain))
            exec_log("virsh shutdown {} {}".format(domain, TAIL))

            data = "vm running"
            while data:
                sleep(3)
                ret = os.popen("virsh list | grep {}".format(domain))
                data = ret.read()
            log.info("{} already shutdown.".format(domain))

            exec_log("virsh start {} {}".format(domain, TAIL))
            reboot_ret = wait_init(ip, port, start_timeout, time_step)
            if reboot_ret:
                log.info("%s start successfully" % domain)
                # wait the system status tend to be stable
                sleep(20)
                return True
            else:
                exec_log("virsh destroy {} {}".format(domain, TAIL))
                sleep(1)
        else:
            exec_log("virsh destroy {} {}".format(domain, TAIL))
            sleep(1)
    return False


def create_vm(para):
    # args: ('win7_sp1_64l_1', '10.1slgfslgf4.24.186', '52:54:00:33:1c:11', '1572864',
    # 'fca2a5fd-f42c-4a62-53f0-9253bde31c11', 'win7_sp1_64l_backup.qcow2')
    domain, ip, mac, memory, uuid, backup_qcow2 = para
    create_qcow2_cmd = "qemu-img create -f qcow2 -b {} {}.qcow2 {}".format(backup_qcow2, domain, TAIL)
    exec_log(create_qcow2_cmd)

    create_xml(domain, uuid, memory, mac)
    log.info("create {}.xml".format(domain))

    exec_log("virsh define {}.xml {}".format(domain, TAIL))

    exec_log("virsh start {} {}".format(domain, TAIL))
    sleep(10)
    return
    ret = start_vm(domain, ip)
    if not ret:
        log.error("create {} failed".format(domain))
        return

    exec_log("virsh snapshot-create {} {}".format(domain, TAIL))
    sleep(2)

    dumpxml_cmd = "virsh dumpxml --migratable {0} > {1}{0}.xml".format(domain, BACKUP_PATH)
    dumpsnapshot_cmd = "virsh snapshot-current {0} > {1}{0}_snapshot.xml".format(domain, BACKUP_PATH)
    exec_log(dumpxml_cmd)
    exec_log(dumpsnapshot_cmd)
    sleep(1)

    exec_log("cp {0}.qcow2 {1}".format(domain, BACKUP_PATH))
    sleep(2)

    exec_log("virsh destroy {} {}".format(domain, TAIL))
    sleep(2)


class WorkManager(object):
    def __init__(self, thread_num=4):
        self.work_queue = Queue.Queue()
        self.threads = []
        self.__init_work_queue()
        self.__init_thread_pool(thread_num)

    def __init_thread_pool(self, thread_num):
        for i in range(thread_num):
            self.threads.append(Work(self.work_queue))

    def __init_work_queue(self):
        for vm in VMS:
            # 任务入队，Queue内部实现了同步机制
            self.work_queue.put((create_vm, vm))

    def wait_allcomplete(self):
        for item in self.threads:
            if item.isAlive():
                item.join()


class Work(threading.Thread):
    def __init__(self, work_queue):
        threading.Thread.__init__(self)
        self.work_queue = work_queue
        self.start()

    def run(self):
        # 死循环，从而让创建的线程在一定条件下关闭退出
        while True:
            try:
                do, args = self.work_queue.get(block=False)  # 任务异步出队，Queue内部实现了同步机制
                do(args)
                self.work_queue.task_done()  # 通知系统任务完成
            except:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--create", help="create vms", action="store_true", required=False)
    parser.add_argument("--define", help="define vms", action="store_true", required=False)
    parser.add_argument("--undefine", help="undefine vms", action="store_true", required=False)
    parser.add_argument("--destroy", help="destroy vms", action="store_true", required=False)
    parser.add_argument("--list_snapshot", help="list snapshot", action="store_true", required=False)
    parser.add_argument("--rm_vm", help="remove vm", action="store_true", required=False)
    parser.add_argument("--test_vm", help="test vm", action="store_true", required=False)
    parser.add_argument("--kvm_conf", help="generate kvm.conf", action="store_true", required=False)
    parser.add_argument("--gen_rem", help="generate rem hostname ip mac", action="store_true", required=False)
    parser.add_argument("--domain", type=str, help="domain", action="store", required=False, default=None)
    parser.add_argument("-p", "--parallel", type=int, help="number of parallel threads", action="store", required=False,
                        default=4)
    args = parser.parse_args()

    os.chdir(XML_PATH)

    vm_object = VmOperation()
    vm_object.get_vm_ip_by_mac()
    for vm in VMS:
        print(vm)

    if args.define:
        vm_object.define()

    if args.undefine:
        vm_object.undefine()

    if args.destroy:
        vm_object.destroy()

    if args.list_snapshot:
        vm_object.list_snapshot(args.domain)

    if args.rm_vm:
        vm_object.rm_vm()

    if args.test_vm:
        vm_object.test_vm()

    if args.create:
        if not os.path.exists(BACKUP_XML_PATH):
            sys.exit(0)
        if not os.path.exists(BACKUP_PATH):
            os.makedirs(BACKUP_PATH)

        start = time()
        work_manager = WorkManager(args.parallel)
        work_manager.wait_allcomplete()
        end = time()
        log.info("cost time: %s seconds." % (end - start))

    if args.kvm_conf:
        generate_kvm_conf()

    if args.gen_rem:
        vm_object.gen_rem_ip_mac()
        for line in DOMAIN_IP_MAC:
            print line
