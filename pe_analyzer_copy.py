#!/usr/bin/python
# coding=utf-8

from random import sample
import csv
from base_tools import *
from sklearn.model_selection import train_test_split

root_dir = "/data/root/pe_classify/"
pefile_dir = os.path.join(root_dir, 'all_pefile')
asm_dir = os.path.join(root_dir, 'all_asm')
report_dir = '/data/root/cuckoo/storage/analyses/'
signature_dir = '/data/root/signatures/'
train_csv = os.path.join(root_dir, '2017game_train.csv')

pred_csv = os.path.join(root_dir, 'pred.csv')
new_label_path = os.path.join(root_dir, 'new_label.npz')
op_path = os.path.join(root_dir, 'op.npz')
api_order_path = os.path.join(root_dir, 'api_order.npz')
imp_path = os.path.join(root_dir, 'imp.npz')
pe_path = os.path.join(root_dir, 'pe.npz')
pix_path = os.path.join(root_dir, 'pix.npz')
api_path = os.path.join(root_dir, 'api.npz')
sig_path = os.path.join(root_dir, 'sig.npz')
sec_path = os.path.join(root_dir, 'sec.npz')
sec_all_path = os.path.join(root_dir, 'sec_all.npz')
sec_new_path = os.path.join(root_dir, 'sec_new.npz')
max_pred = os.path.join(root_dir, 'max.npz')
max_pred1 = os.path.join(root_dir, 'max1.npz')

BOUNDARY = '; ---------------------------------------------------------------------------'
MAXLEN = 10000
OUTPUT_DIM = 50
INPUT_DIM = 0
MAX_NB_WORDS = 0

log = logging.getLogger(os.path.splitext(__file__)[0])


def get_total_sig():
    def get_sig_name(py):
        with open(py) as f:
            sig_line = filter(lambda line: line.startswith('    name = '), f.readlines())
            sig_name = map(lambda s: s.strip().split('"')[1], sig_line)
            return sig_name

    os.chdir(signature_dir)
    all_sig_names = map(get_sig_name, os.listdir(signature_dir))
    return sum(all_sig_names, [])


def isvalid(s):
    bytes = '0123456789abcdef'
    if len(s) == 2:
        if s[0] in bytes:
            return False  # ins cannot have these
    if not s.isalpha():
        return False
    if s[0].isupper():
        return False
    if s in ['align', 'extrn', 'unicode', 'assume', 'offset']:
        return False
    return True


def get_op(md5):
    opcodes = []
    asm_path = md5 + '.asm'
    if os.path.exists(asm_path):
        with open(md5 + '.asm', 'r') as f:
            for line in f:
                if not line.startswith('\t'):
                    continue
                try:
                    opcode = line.strip().split()[0]
                except IndexError:
                    continue
                if isvalid(opcode):
                    opcodes.append(opcode)
                    if len(opcodes) >= MAXLEN:
                        break

    return opcodes


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(input_list, token_indice, ngram_range):
    new_list = []
    for ngram_value in range(2, ngram_range + 1):
        for i in range(len(input_list) - 1):
            ngram = tuple(input_list[i:i + ngram_value])
            if ngram in token_indice and len(ngram) >= ngram_value:
                new_list.append(token_indice[ngram])
    return input_list + new_list


def op2int(ops_list_list):
    all_ops = list(reduce_list_list(ops_list_list))
    log.info('length of all_ops: %s', len(all_ops))
    token_indice = {v: k for k, v in enumerate(all_ops)}

    def _encode(ops_list):
        _op_int = []
        for op in ops_list:
            if op in token_indice:
                _op_int.append(token_indice[op])
        return _op_int

    return map(_encode, ops_list_list)


def multi_encoding_op(ops_list_list, ngram_range=2, filter_count=1):
    op_int = op2int(ops_list_list)

    assert ngram_range > 1, 'ngram_range must be larger than 1!'

    # log.info('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    # if ngram_range = 2, then [1, 4, 9, 4, 1, 4] --> {(4, 9), (4, 1), (1, 4), (9, 4)}
    log.debug('create ngram_sets')
    ngram_set = set()
    for input_list in op_int:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order to avoid collision with existing features.
    log.debug('create token_indice')
    max_features = 10000
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1
    log.info('max_features: %s', max_features)

    # token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    # [[1, 3, 4, 5], [1, 3, 7, 9, 2]] --> [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    log.debug('add_ngram')
    add_ngram_p = functools.partial(add_ngram, token_indice=token_indice, ngram_range=ngram_range)
    op_encode_l = pool_map(add_ngram_p, op_int)

    log.debug('encode opcodes')
    return multi_encoding_l_c(op_encode_l, filter_count)


def get_imp_name(target):
    imp_names = set()
    try:
        pe = pefile.PE(target)
    except:
        log.warning("%s is not valid PE File" % target)
        return imp_names

    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                imp_names.add(imp.name)
    return imp_names


def get_md5_from_select(target):
    return target[0].rsplit('/', 1)[-1]


def get_pe_header(md5):
    try:
        pe = pefile.PE(md5)
        NumberOfSections = pe.FILE_HEADER.NumberOfSections
        NumberOfSymbols = pe.FILE_HEADER.NumberOfSymbols
        SizeOfOptionalHeader = pe.FILE_HEADER.SizeOfOptionalHeader
        SizeOfCode = pe.OPTIONAL_HEADER.SizeOfCode
        SizeOfInitializedData = pe.OPTIONAL_HEADER.SizeOfInitializedData
        SizeOfUninitializedData = pe.OPTIONAL_HEADER.SizeOfUninitializedData
        SizeOfImage = pe.OPTIONAL_HEADER.SizeOfImage
        SizeOfHeaders = pe.OPTIONAL_HEADER.SizeOfHeaders
        SizeOfStackReserve = pe.OPTIONAL_HEADER.SizeOfStackReserve
        SizeOfStackCommit = pe.OPTIONAL_HEADER.SizeOfStackCommit
        SizeOfHeapReserve = pe.OPTIONAL_HEADER.SizeOfHeapReserve
        SizeOfHeapCommit = pe.OPTIONAL_HEADER.SizeOfHeapCommit
        result = [NumberOfSections, NumberOfSymbols, SizeOfOptionalHeader, SizeOfCode, SizeOfInitializedData,
                  SizeOfUninitializedData, SizeOfImage, SizeOfHeaders, SizeOfStackReserve, SizeOfStackCommit,
                  SizeOfHeapReserve, SizeOfHeapCommit]
    except:
        log.warning("%s is not valid PE File" % md5)
        result = [0] * 12
    return result


def multi_get_pe(md5s, pe_dir=pefile_dir):
    os.chdir(pe_dir)
    return pool_map(get_pe_header, md5s)


@log_decorate
def multi_get_md5s(start_id=1, end_id=total_count):
    conn = MySQLdb.connect(host='localhost', port=3306, user='root', passwd='root123', db='sandbox')
    cur = conn.cursor()
    _targets = cur.execute('select target from tasks where id >= {} and id <= {}'.format(start_id, end_id))
    targets = cur.fetchmany(_targets)
    cur.close()
    conn.close()

    return pool_map(get_md5_from_select, targets)


def multi_get_op(md5s, op_dir=asm_dir):
    os.chdir(op_dir)
    return pool_map(get_op, md5s)


def multi_get_imp(md5s, imp_dir=pefile_dir):
    os.chdir(imp_dir)
    return pool_map(get_imp_name, md5s)


def get_api(i):
    report_path = os.path.join(report_dir, str(i), 'reports/report.json')
    api_str = ''
    apistats_flag = False
    api_dict = Counter()
    if os.path.exists(report_path):
        with open(report_path) as f:
            f.seek(900)
            for line in f:
                if line.startswith('        "apistats":'):
                    api_str = '{'
                    apistats_flag = True
                    continue
                if line.startswith('        "processes":'):
                    api_str = api_str[:-2]
                    break
                if apistats_flag:
                    api_str += line[:-1]

    if apistats_flag:
        _api_dict = json.loads(api_str)
        api_dict = reduce(lambda c1, c2: c1 + c2, map(lambda x: Counter(x), _api_dict.values()))
    return api_dict


def multi_get_api(start_id=1, end_id=total_count):
    return pool_map(get_api, range(start_id, end_id + 1))


def get_order_api(i):
    report_path = os.path.join(report_dir, str(i), 'reports/report.json')
    api_list = []
    if os.path.exists(report_path):
        with open(report_path) as f:
            f.seek(900)
            for line in f:
                if line.startswith('                        "api":'):
                    line_strip = line.strip()
                    sig = line_strip.split('"')[3]
                    api_list.append(sig)
    return api_list


def multi_get_order_api(start_id=1, end_id=total_count):
    return pool_map(get_order_api, range(start_id, end_id + 1))


def get_sig(i):
    report_path = os.path.join(report_dir, str(i), 'reports/report.json')
    sigs_set = set()
    if os.path.exists(report_path):
        with open(report_path) as f:
            f.seek(900)
            for line in f:
                if line.startswith('            "name":'):
                    line_strip = line.strip()
                    if not line_strip.endswith(','):
                        sig = line_strip.split('"')[3]
                        sigs_set.add(sig)
                if line.startswith('    "target": {'):
                    break

    return sigs_set


def multi_get_sig(start_id=1, end_id=total_count):
    return pool_map(get_sig, range(start_id, end_id + 1))


def get_section(i):
    report_path = os.path.join(report_dir, str(i), 'reports/report.json')
    section_str = ''
    section_flag = False
    section_dict = {}
    if os.path.exists(report_path):
        with open(report_path) as f:
            f.seek(900)
            for line in f:
                if line.startswith('        "pe_sections":'):
                    section_str = '{' + line.strip()
                    section_flag = True
                    continue

                if section_flag:
                    section_str += line[:-1]
                    if line.startswith('    },'):
                        section_str = section_str[:-2]
                        break

    if section_flag:
        section_dict = json.loads(section_str)['pe_sections']

    return section_dict


def multi_get_section(start_id=1, end_id=total_count):
    return pool_map(get_section, range(start_id, end_id + 1))


def get_top_pix(md5, k=1000):
    with open(md5, 'rb') as f:
        content = f.read(k)
    hexst = binascii.hexlify(content)  # 将二进制文件转换为十六进制字符串
    fh = [int(hexst[i: i + 2], 16) for i in range(0, len(hexst), 2)]  # 按字节分割
    if len(fh) < k:
        fh.extend([0] * (k - len(fh)))
    return fh


def multi_get_pix(md5s, pix_dir=pefile_dir):
    os.chdir(pix_dir)
    return pool_map(get_top_pix, md5s)


def train_predict(feature, label, n_estimators, index_train, index_val):
    assert isinstance(label, np.ndarray)
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=4)
    clf.fit(feature[:train_count][list(index_train), :], label[list(index_train)])
    y_val_pred = clf.predict(feature[:train_count][list(index_val), :])
    val_pred_accu = np.mean(y_val_pred == label[list(index_val)])
    y_pred = clf.predict(feature[train_count:])
    return y_val_pred, val_pred_accu, y_pred


def train_predict_full(feature, label, n_estimators):
    assert isinstance(label, np.ndarray)
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=4)
    clf.fit(feature[:train_count], label)
    y_pred = clf.predict(feature[train_count:])
    return y_pred


def most_common(x):
    x2 = Counter(x).most_common(1)
    return x2[0][0]


def sec2name(d_list, sec_k):
    if d_list:
        return reduce(lambda d1, d2: dict(d1, **d2), [{d['name']: d[sec_k]} for d in d_list])
    else:
        return {}


if __name__ == "__main__":
    log.info('#################################################################')
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", help="display debug messages", action="store_true", required=False)
    parser.add_argument("--op", help="get opcode from asm", action="store_true", required=False)
    parser.add_argument("--imp", help="get import function from pefile", action="store_true", required=False)
    parser.add_argument("--pe", help="get pe header from pefile", action="store_true", required=False)
    parser.add_argument("--pix", help="get topk pix from pefile", action="store_true", required=False)
    parser.add_argument("--api", help="get api from report", action="store_true", required=False)
    parser.add_argument("--sig", help="get signature from report", action="store_true", required=False)
    parser.add_argument("--api1", help="get ordered api from report", action="store_true", required=False)
    parser.add_argument("--sec", help="get sections from report", action="store_true", required=False)
    parser.add_argument("--rf", help="random forest", action="store_true", required=False)
    parser.add_argument("-m", "--most_common", help="get most result from y_pred", action="store_true", required=False)
    args = parser.parse_args()

    # get md5s from mysql
    md5s = multi_get_md5s()
    md5s_df = pd.DataFrame(md5s, columns=['md5'])

    # create empty test csv
    test_label_data = zip(md5s_df['md5'][train_count:], [None] * test_count)
    csv_file = os.path.join(root_dir, 'test.csv')
    csvfile = file(csv_file, 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['md5', 'type'])
    writer.writerows(test_label_data)
    csvfile.close()

    all_label = pd.concat([pd.read_csv(train_csv), pd.read_csv(csv_file)])
    md5_label_api = pd.merge(md5s_df, all_label, 'left', 'md5')

    # transfer type[string] to int
    train_types = set(md5_label_api['type'][:train_count])
    token_indice = {v: k for k, v in enumerate(train_types)}
    y_train_val = np.array([token_indice[i] for i in md5_label_api['type'][:train_count]])

    # get validation index
    index_train, index_val, y_train, y_val = train_test_split(range(0, train_count), y_train_val,
                                                              stratify=y_train_val, random_state=0, test_size=0.2)
    if args.debug:
        log.setLevel(logging.DEBUG)

    if args.op:
        log.debug('process opcodes')
        ops = multi_get_op(md5s)
        ops_encoding = multi_encoding_op(ops, ngram_range=3, filter_count=100)
        log.debug('save op')
        np.savez_compressed(op_path, md5=md5s, op=ops_encoding)
        log.debug('process opcodes finished')

    if args.imp:
        log.debug('process import functions')
        imps = multi_get_imp(md5s)
        imps_encoding = multi_encoding_l_s(imps)
        log.debug('save imp')
        np.savez_compressed(imp_path, md5=md5s, imp=imps_encoding)
        log.debug('process import functions finished')

    if args.pe:
        log.debug('process pe header')
        pes = multi_get_pe(md5s)
        log.debug('save pe')
        np.savez_compressed(pe_path, md5=md5s, pe=pes)
        log.debug('process pe header finished')

    if args.pix:
        log.debug('process pix functions')
        pixs = multi_get_pix(md5s)
        log.debug('save pix')
        np.savez_compressed(pix_path, md5=md5s, pix=pixs)
        log.debug('process pix functions finished')

    if args.api:
        log.debug('process apis')
        apis = multi_get_api()
        apis_encoding = multi_encoding_l_c(apis)
        log.debug('save api')
        np.savez_compressed(api_path, md5=md5s, api=apis_encoding)
        log.debug('process apis finished')

    if args.sig:
        log.debug('process signatures')
        sigs = multi_get_sig()
        sigs_encoding = multi_encoding_l_s(sigs)
        log.debug('save sig')
        np.savez_compressed(sig_path, md5=md5s, sig=sigs_encoding)
        log.debug('process signatures finished')

    if args.api1:
        log.debug('process ordered apis')
        order_apis = multi_get_order_api()
        log.debug('process pickling ordered apis')
        order_api_p_path = os.path.join(root_dir, 'order_api.pkl')
        with open(order_api_p_path, 'wb') as f:
            cPickle.dump(order_apis, f)

        log.debug('process encoding ordered apis')
        order_ops_encoding = multi_encoding_op(order_apis, ngram_range=3, filter_count=2)
        log.debug('save ordered op')
        np.savez_compressed(api_order_path, md5=md5s, op=order_ops_encoding)
        log.debug('process order opcodes finished')

    if args.sec:
        log.debug('process sections')
        # sec_keys = ['entropy', 'size_of_data', 'virtual_address', 'virtual_size']
        sec_keys = ['entropy']
        secs = multi_get_section()
        secs_encoding = np.array([])
        for s_k in sec_keys:
            sec2name_p = functools.partial(sec2name, sec_k=s_k)
            section_l_d = map(sec2name_p, secs)
            _secs_encoding = multi_encoding_l_c(section_l_d, filter_count=1)
            if len(secs_encoding):
                secs_encoding = np.concatenate([secs_encoding, np.array(_secs_encoding)], axis=1)
            else:
                secs_encoding = _secs_encoding

        log.debug('save sec')
        np.savez_compressed(sec_new_path, md5=md5s, sec=secs_encoding)
        log.debug('process secs finished')

    if args.rf:
        ops = np.load(op_path)['op']
        imps = np.load(imp_path)['imp']
        pixs = np.load(pix_path)['pix']
        apis = np.load(api_path)['api']
        # sigs = np.load(sig_path)['sig']
        secs = np.load(sec_path)['sec']
        pes = np.load(pe_path)['pe']

        ks = ['ops', 'imps', 'pes', 'pixs', 'apis', 'secs']
        vs = [ops, imps, pes, pixs, apis, secs]
        f_dict = OrderedDict(zip(ks, vs))
        all_f = sum(map(lambda x: list(x), [combinations(f_dict, i) for i in range(2, len(ks) + 1)]), [])
        n_estimators = 100
        log.info('running random forest with %s estimators', n_estimators)

        # TODO: to delete
        test_csv = os.path.join(root_dir, '2017game_test.csv')
        _all_label = pd.concat([pd.read_csv(train_csv), pd.read_csv(test_csv)])
        _md5_label_api = pd.merge(md5s_df, _all_label, 'left', 'md5')
        # transfer type[string] to int
        _type_encode = np.array([token_indice[i] for i in _md5_label_api['type'][train_count:]])

        y_pred_accu_all = []
        for n_estimators in [10, 20, 50, 100, 200, 500, 1000]:
            for k in ['pixs']:
                pred_path = os.path.join(root_dir, 'npzs', k + '_pred.npz')
                # if os.path.exists(pred_path):
                #     log.debug('skip %s.', pred_path)
                #     continue
                y_pred = train_predict_full(f_dict[k], y_train_val, n_estimators)
                y_pred_accu = np.mean(np.array(y_pred) == _type_encode)
                # np.savez_compressed(pred_path, pred=y_pred)
                log.debug('feature: %s, validation pred_accu: %.5f' % (k, y_pred_accu))
                y_pred_accu_all.append(y_pred_accu)
        log.info(y_pred_accu_all)
        sys.exit()

        for fea in all_f:
            pred_path = os.path.join(root_dir, 'npzs', '_'.join(fea) + '_pred.npz')
            if os.path.exists(pred_path):
                log.debug('skip %s.', pred_path)
                continue

            fea_array = map(lambda _f: f_dict[_f], fea)
            _fea = np.concatenate(fea_array, axis=1)
            y_pred = train_predict_full(_fea, y_train_val, n_estimators)
            y_pred_accu = np.mean(np.array(y_pred) == _type_encode)
            del _fea, fea_array
            gc.collect()
            np.savez_compressed(pred_path, pred=y_pred)
            log.info('combined features: %s finished', fea)
            log.info('combined features: %s, max pred_accu: %.5f' % (fea, y_pred_accu))

    if args.most_common:
        os.chdir(root_dir)
        # ks = ['ops', 'imps', 'pixs', 'apis', 'sigs', 'secs']
        ks = ['ops', 'imps', 'pes', 'pixs', 'apis', 'secs']
        all_ks = sum(map(lambda x: list(x), [combinations(ks, i) for i in range(2, len(ks) + 1)]), [])
        val_npzs = map(lambda k: '_'.join(k) + '_val_pred.npz', all_ks)

        max_accu = 0.73
        com_npz = ()
        y_val_true = y_val
        log.debug('load pred data finished.')

        # TODO: to delete
        test_csv = os.path.join(root_dir, '2017game_test.csv')
        _all_label = pd.concat([pd.read_csv(train_csv), pd.read_csv(test_csv)])
        _md5_label_api = pd.merge(md5s_df, _all_label, 'left', 'md5')
        # transfer type[string] to int
        _type_encode = np.array([token_indice[i] for i in _md5_label_api['type'][train_count:]])

        val_npzs_data = map(lambda a: (a, np.load(a)['pred']), val_npzs)
        sample_num = 5
        iter_number = 500
        log.info('iteration number: %s', iter_number)
        for sample_num in range(3, 30):
            y_pred_accu_val_all = []
            y_pred_accu_test_all = []
            log.info('sample number: %s', sample_num)
            for i in range(iter_number):
                j = sample(val_npzs_data, sample_num)
                y_combined = [z[1] for z in j]
                y_pred_c = pool_map(most_common, np.array(y_combined).T)
                y_pred_accu = np.mean(np.array(y_pred_c) == y_val_true)
                y_pred_accu_val_all.append(y_pred_accu)
                if 1:  # y_pred_accu > max_accu:
                    # max_accu = y_pred_accu
                    val_com_npz = map(lambda x: x[0], j)
                    log.debug('tmp maximun val pred_accu: %.5f, combined npz: %s' % (y_pred_accu, val_com_npz))

                    test_com_npz = map(lambda x: x.replace('_val', ''), val_com_npz)
                    test_npzs_data = map(lambda a: (a, np.load(os.path.join(root_dir, 'npzs', a))['pred']),
                                         test_com_npz)
                    y_combined = [z[1] for z in test_npzs_data]
                    y_pred_c = pool_map(most_common, np.array(y_combined).T)

                    y_pred_accu = np.mean(np.array(y_pred_c) == _type_encode)
                    y_pred_accu_test_all.append(y_pred_accu)
                    log.debug('tmp test pred_accu: %.5f, combined npz: %s\n\n' % (y_pred_accu, test_com_npz))
            log.info('ave accu val:%.5f, test: %.5f' % (np.mean(y_pred_accu_val_all), np.mean(y_pred_accu_test_all)))
            log.info('max accu val:%.5f, test: %.5f' % (np.max(y_pred_accu_val_all), np.max(y_pred_accu_test_all)))
        sys.exit()


        def _predict(npz_l):
            # ('ops_apis_secs_val_pred.npz', 'ops_imps_val_pred.npz', 'apis_val_pred.npz', 'imps_secs_val_pred.npz')
            _npz_data = map(lambda a: np.load(a)['pred'], npz_l)
            _y_pred_c = pool_map(most_common, np.array(_npz_data).T)
            return _y_pred_c


        val_73_npz_path = os.path.join(root_dir, 'val_73_npz.json')
        with open(val_73_npz_path, 'r') as f:
            val_73_npz = json.load(f)['val_73_npz']

        for i in range(0):
            val_com_npz = sample(val_73_npz, sample_num)
            if not i % 2000:
                log.debug('i: %s', i)

            os.chdir(root_dir)
            y_combined = map(_predict, val_com_npz)

            max_accu = 0.0
            y_pred_c = pool_map(most_common, np.array(y_combined).T)
            y_pred_accu = np.mean(np.array(y_pred_c) == y_val_true)
            if y_pred_accu > max_accu:
                # max_accu = y_pred_accu
                log.debug('tmp maximun val pred_accu: %.5f, combined npz: %s' % (y_pred_accu, val_com_npz))

                os.chdir(os.path.join(root_dir, 'npzs'))
                test_com_npz = [map(lambda x: x.replace('_val', ''), npz) for npz in val_com_npz]

                y_combined = map(_predict, test_com_npz)
                y_pred_c = pool_map(most_common, np.array(y_combined).T)
                # TODO: to delete
                y_pred_accu = np.mean(np.array(y_pred_c) == _type_encode)
                log.debug('tmp test pred_accu: %.5f, combined npz: %s\n\n' % (y_pred_accu, test_com_npz))
