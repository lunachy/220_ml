# coding=utf-8

ddl1 = "CREATE TABLE `{}` ("
ddl2 = """  `uuid` int(11) NOT NULL AUTO_INCREMENT,
  `attack_date` datetime DEFAULT NULL ON UPDATE current_timestamp() COMMENT '攻击时间',
  `assert_id` varchar(255) CHARACTER SET utf8 DEFAULT NULL COMMENT '资产id',
  `attack_cnt` int(11) DEFAULT NULL COMMENT '攻击次数',
  `Predict` varchar(255) CHARACTER SET utf8 DEFAULT NULL COMMENT '预测值',"""
ddl3 = "  `{}` varchar(255) CHARACTER SET utf8 DEFAULT NULL COMMENT '{}',"
ddl4 = """  PRIMARY KEY (`uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;"""


cal_table_names='password_guess_cal_info,web_attack_cal_info,malicious_scan_cal_info,malicious_program_cal_info,ddos_attack_cal_info,log_crack_cal_info,system_auth_cal_info,error_log_cal_info,vul_used_cal_info,configure_compliance_cal_info,weak_password_cal_info'
cal_tables = cal_table_names.split(',')

with open('create_cal_tables_ddl.txt', 'w') as f:
    for cal_t in cal_tables:
        f.write(ddl1.format(cal_t) + '\n')
        f.write(ddl2 + '\n')
        for line in open("field.txt"):
            name, comment = filter(lambda x: x, line.strip().split(' '))
            # print comment, name
            f.write(ddl3.format(name, comment) + '\n')
        f.write(ddl4 + '\n\n')
