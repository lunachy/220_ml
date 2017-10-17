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
op_path = os.path.join(root_dir, 'op.npz')
api_order_path = os.path.join(root_dir, 'api_order.npz')
imp_path = os.path.join(root_dir, 'imp.npz')
pe_path = os.path.join(root_dir, 'pe.npz')
pix_path = os.path.join(root_dir, 'pix.npz')
api_path = os.path.join(root_dir, 'api.npz')
sig_path = os.path.join(root_dir, 'sig.npz')
sec_path = os.path.join(root_dir, 'sec.npz')
sec_new_path = os.path.join(root_dir, 'sec_new.npz')
sec_all_path = os.path.join(root_dir, 'sec_all.npz')

BOUNDARY = '; ---------------------------------------------------------------------------'
MAXLEN = 10000

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


# @log_decorate
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
        # print("%s, not valid PE File" % target)
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
        # Dll = pe.OPTIONAL_HEADER.DllCharacteristics
        SizeOfStackReserve = pe.OPTIONAL_HEADER.SizeOfStackReserve
        SizeOfStackCommit = pe.OPTIONAL_HEADER.SizeOfStackCommit
        SizeOfHeapReserve = pe.OPTIONAL_HEADER.SizeOfHeapReserve
        SizeOfHeapCommit = pe.OPTIONAL_HEADER.SizeOfHeapCommit
        result = [NumberOfSections, NumberOfSymbols, SizeOfOptionalHeader, SizeOfCode, SizeOfInitializedData,
                  SizeOfUninitializedData, SizeOfImage, SizeOfHeaders, SizeOfStackReserve, SizeOfStackCommit,
                  SizeOfHeapReserve, SizeOfHeapCommit]
    except:
        log.warning("%s, not valid PE File" % md5)
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
    hexst = binascii.hexlify(content)  # bin to hex
    fh = [int(hexst[i: i + 2], 16) for i in range(0, len(hexst), 2)]
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
    return y_val_pred, val_pred_accu


def train_predict_importance(feature, label, n_estimators, index_train, index_val):
    assert isinstance(label, np.ndarray)
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=4)
    clf.fit(feature[:train_count][list(index_train), :], label[list(index_train)])
    importances = clf.feature_importances_
    ave = np.sum(importances >= np.mean(importances))
    idx = np.argsort(importances)[::-1][0:ave]
    return idx


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


def write_test_csv(csv_file, data):
    csvfile = file(csv_file, 'wb')
    writer = csv.writer(csvfile)
    writer.writerow(['md5', 'type'])
    writer.writerows(data)
    csvfile.close()


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

    test_label_data = zip(md5s_df['md5'][train_count:], [None] * test_count)
    csv_file = os.path.join(root_dir, 'test.csv')
    write_test_csv(csv_file, test_label_data)

    all_label = pd.concat([pd.read_csv(train_csv), pd.read_csv(csv_file)])
    md5_label_api = pd.merge(md5s_df, all_label, 'left', 'md5')

    # transfer type[string] to int
    train_types = set(md5_label_api['type'][:train_count])
    token_indice = {v: k for k, v in enumerate(train_types)}
    indice_token = {k: v for k, v in enumerate(train_types)}
    y_train_val = np.array([token_indice[i] for i in md5_label_api['type'][:train_count]])

    # get validation index
    index_train, index_val, y_train, y_val = train_test_split(range(train_count), y_train_val,
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
        sec_keys = ['entropy', 'size_of_data', 'virtual_address', 'virtual_size']
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
        np.savez_compressed(sec_all_path, md5=md5s, sec=secs_encoding)
        log.debug('process secs finished')

    if args.rf:
        ops = np.load(op_path)['op']
        imps = np.load(imp_path)['imp']
        pixs = np.load(pix_path)['pix']
        apis = np.load(api_path)['api']
        # sigs = np.load(sig_path)['sig']
        secs = np.load(sec_new_path)['sec']
        pes = np.load(pe_path)['pe']

        # choose important features from ops and imps
        n_estimators = 200
        idx_ops = train_predict_importance(ops, y_train_val, n_estimators, index_train, index_val)
        idx_imps = train_predict_importance(imps, y_train_val, n_estimators, index_train, index_val)
        opsS = ops[:, idx_ops]
        impsS = imps[:, idx_imps]
        log.debug('important rf factors, ops: %s, imps: %s' % (len(idx_ops), len(idx_imps)))

        ks = ['opsS', 'impsS', 'pes', 'pixs', 'apis', 'secsS']
        vs = [opsS, impsS, pes, pixs, apis, secs]
        f_dict = OrderedDict(zip(ks, vs))
        all_f = sum(map(lambda x: list(x), [combinations(f_dict, i) for i in range(1, len(ks) + 1)]), [])
        log.info('running random forest with %s estimators', n_estimators)

        # train the validation data
        for k in all_f:
            if len(k) == 1:  # ('opsS',)...
                pred_val_path = os.path.join(root_dir, k[0] + '_val_pred.npz')
                _fea = f_dict[k[0]]
            else:  # combinations of ['opsS', 'impsS', 'pes', 'pixs', 'apis', 'secsS']
                pred_val_path = os.path.join(root_dir, '_'.join(k) + '_val_pred.npz')
                fea_array = map(lambda _k: f_dict[_k], k)
                _fea = np.concatenate(fea_array, axis=1)

            max_y_accu = 0.0
            max_y_val_pred = []
            for _ in range(3):
                y_val_pred, val_pred_accu = train_predict(_fea, y_train_val, n_estimators, index_train, index_val)
                if val_pred_accu > max_y_accu:
                    max_y_accu = val_pred_accu
                    max_y_val_pred = y_val_pred
            np.savez_compressed(pred_val_path, pred=max_y_val_pred)
            log.info('feature: %s, max val pred_accu: %.5f' % (k, max_y_accu))

        # train the full data
        for k in all_f:
            if len(k) == 1:  # ('opsS',)...
                pred_path = os.path.join(root_dir, 'npzs', k[0] + '_pred.npz')
                _fea = f_dict[k[0]]
            else:  # combinations of ['opsS', 'impsS', 'pes', 'pixs', 'apis', 'secsS']
                pred_path = os.path.join(root_dir, 'npzs', '_'.join(k) + '_pred.npz')
                fea_array = map(lambda _k: f_dict[_k], k)
                _fea = np.concatenate(fea_array, axis=1)

            y_pred = train_predict_full(_fea, y_train_val, n_estimators)
            np.savez_compressed(pred_path, pred=y_pred)
        log.info('test predict finished.')

    if args.most_common:
        os.chdir(root_dir)
        ks = ['opsS', 'impsS', 'pes', 'pixs', 'apis', 'secsS']
        all_ks = sum(map(lambda x: list(x), [combinations(ks, i) for i in range(2, len(ks) + 1)]), [])
        val_npzs = map(lambda k: '_'.join(k) + '_val_pred.npz', all_ks)
        thresh_accu = 0.73
        sample_num = 6
        thresh_num = 100
        com_npz = tuple()
        val_y_true = y_val
        val_npzs_data = map(lambda a: (a, np.load(a)['pred']), val_npzs)
        log.debug('sample number: %s, thresh_num: %s' % (sample_num, thresh_num))

        # first integration, filter those whose validation accuracy is less than thresh_accu
        filter_val_com_npz = []

        while len(filter_val_com_npz) < thresh_num:
            j = sample(val_npzs_data, sample_num)
            val_y_combined = [z[1] for z in j]
            val_y_pred_c = pool_map(most_common, np.array(val_y_combined).T)
            val_y_pred_accu = np.mean(np.array(val_y_pred_c) == val_y_true)
            if val_y_pred_accu > thresh_accu:
                val_com_npz = map(lambda x: x[0], j)
                filter_val_com_npz.append(val_com_npz)
                log.debug('val pred_accu: %.5f, combined npz: %s' % (val_y_pred_accu, val_com_npz))


        def _predict(npz_l):
            # ('ops_apis_secs_val_pred.npz', 'ops_imps_val_pred.npz', 'apis_val_pred.npz', 'imps_secs_val_pred.npz')
            _npz_data = map(lambda a: np.load(a)['pred'], npz_l)
            _y_pred = pool_map(most_common, np.array(_npz_data).T)
            return _y_pred


        # second integration, predict the test dataset
        iter_num = 100
        os.chdir(os.path.join(root_dir, 'npzs'))
        for i in range(iter_num):
            val_com_npz = sample(filter_val_com_npz, sample_num)
            test_com_npz = [map(lambda x: x.replace('_val', ''), npz) for npz in val_com_npz]
            test_y_combined = map(_predict, test_com_npz)
            test_y_pred = pool_map(most_common, np.array(test_y_combined).T)
            test_label = map(lambda x: indice_token[x], test_y_pred)

            test_label_data = zip(md5s_df['md5'][train_count:], test_label)
            csv_file = os.path.join(root_dir, 'test_csv', str(i) + '.csv')
            write_test_csv(csv_file, test_label_data)
