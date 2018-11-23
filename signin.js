//console.show();
const app_t = 15000;
const click_t = 8000;
const back_t = 1000;
const t = click_t;
images.requestScreenCapture();

function signin(app, app_cos, signin_cos) {
    home();
    sleep(back_t);
    swipe(900, 1000, 200, 1000, 100);
    sleep(back_t);

    if (app === 'com.eg.android.AlipayGphone') {
        launch(app);
    }
    else {
        for (let i = 0; i < app_cos.length; i++) {
            click(app_cos[i][0], app_cos[i][1]);
            sleep(back_t);
        }
    }
    sleep(app_t);

    for (let i = 0; i < signin_cos.length; i++) {
        click(signin_cos[i][0], signin_cos[i][1]);
        sleep(click_t);
    }
    if (app === 'com.eg.android.AlipayGphone') {
        let count = 10;
        while (count > 0) {
            click(538, 530);
            sleep(500);
            count--;
        }
    }

    //images.captureScreen('/sdcard/脚本/signin/' + app + '.jpg');

    for (let i = 0; i < signin_cos.length + 2; i++) {
        back();
        sleep(back_t);
    }
}

function unlock_screen() {
    max_retry_times = 3;
    let screen_on = device.isScreenOn();
    while (!screen_on && max_retry_times) {
        device.wakeUp();
        sleep(back_t);
        swipe(100, 1900, 100, 400, 100);
        sleep(back_t);
        let pass = [[540, 1675], [220, 1455], [860, 1675], [860, 1455], [220, 1455]];
        pass.map(e => click(e[0], e[1]));

        sleep(back_t * 12);
        screen_on = device.isScreenOn();
        max_retry_times -= 1;
    }

    if (max_retry_times) {
        log("signin解锁成功");
    } else {
        log("signin解锁失败");
        let time = new Date();
        images.captureScreen('/sdcard/脚本/signin/' + time.getMinutes() + '_' + time.getSeconds() + '.jpg');
        return false;
    }
    return true;
}

if (unlock_screen()) {
    signin('com.UCMobile', [[950, 2100]], [[980, 2200], [848, 306]]);

    signin('com.baidu.BaiduMap', [[400, 200], [300, 800]], [[126, 171], [837, 311], [486, 691]]);
    signin('com.gtgj.view', [[400, 200], [550, 800]], [[964, 2186], [915, 300], [480, 1870], [528, 1290]]);
    signin('com.unionpay', [[400, 200], [800, 800]], [[990, 156], [500, 900], [550, 770]]);
    signin('com.youku.phone', [[400, 200], [300, 1100]], [[1000, 2200], [885, 304]]);
    signin('com.qiyi.video', [[400, 200], [550, 1100]], [[767, 2200], [885, 304]]);
    //signin('com.tencent.qqlive', [[400, 200], [800, 1100]], [[985, 2200], [885, 376]]);
    //signin('com.mfw.roadbook', [[400, 200], [300, 1400]], [[985, 2191], [884, 1248]]);
    signin('com.jingdong.app.mall', [[400, 200], [550, 1400]], [[976, 2200], [135, 893], [538, 507], [550, 380], [280, 1175]]);
    //signin('com.suning.mobile.ebuy', [[400, 200], [800, 1400]], [[976, 2200], [192, 553], [902, 357]]);

    //signin('com.jd.jrapp', [[900, 500], [200, 1100]], [[961, 2208], [184, 695], [818, 600]]);
    //signin('com.baidu.searchbox', [[900, 500], [550, 1100]], [[974, 2229], [977, 227]]);
    
    
    signin('com.netease.cloudmusic', [[150, 200], [300, 1400]], [[79, 171], [750, 441]]);
    signin('fm.qingting.qtradio', [[150, 200], [550, 1400]], [[970, 2189], [947, 330]]);

    signin('com.eg.android.AlipayGphone', [], [[964, 2186], [539, 510], [155, 1000]]);
}