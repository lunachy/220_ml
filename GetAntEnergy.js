// auto();
// console.show()
// Y/X   220 540 860
// 1455   1   2   3
// 1675   4   5   6
// 1885   7   8   9
// 2100       0
const handimg = images.read('/sdcard/脚本/take.png');
const gimg = images.read('/sdcard/脚本/g.jpg');
let [W, H] = [device.width, device.height];

const alipayHome = className("android.widget.TextView").text("首页");
const antIcon = className("android.widget.TextView").text("蚂蚁森林");
const hezhong = className("android.widget.Button").desc("合种");
const myEnergyCat = className("android.view.View").descMatches(/线下支付|行走|生活缴费/);
const energy = className("android.widget.Button").descStartsWith("收集能量");
const canget = className("android.view.View").descEndsWith("可收取");
const moreFriend = className('android.view.View').desc("查看更多好友");
const noMore = className('android.view.View').desc("没有更多了");
const friendRank = className("android.widget.TextView").text("好友排行榜");
const inforest = className("android.widget.TextView").textEndsWith("的蚂蚁森林");
const alipay_package = "com.eg.android.AlipayGphone";

function click_widget(widget) {
    rect = widget.bounds();
    click(rect.centerX(), rect.centerY());
}

const timeout = 1000;
let max_retry_times = 5;
images.requestScreenCapture();

function enter_ant_forest() {
    max_retry_times = 5;
    //click(680, 1080);
    //sleep(timeout);
    launch(alipay_package); //启动支付宝
    sleep(timeout * 3);
    click(945, 530);    //蚂蚁森林坐标，防止点击控件失败
    //click(930, 976);
    sleep(timeout * 5);
    // let home = alipayHome.findOne(timeout);
    // while (!home && max_retry_times) {
    //     log("进入支付宝首页失败，可能在支付宝别的界面，按返回键直到首页");
    //     back();
    //     max_retry_times -= 1;
    //     sleep(timeout);
    //     home = alipayHome.findOne(timeout);
    // }
    // if (max_retry_times) {
    //     log("进入支付宝首页成功");
    // } else {
    //     log("进入支付宝首页失败");
    //     let time = new Date();
    //     images.captureScreen('/sdcard/脚本/' + time.getMinutes() + '_' + time.getSeconds() + '.jpg');
    //     return false;
    // }
    //
    // let ant_forest = antIcon.findOne(timeout);
    // if (!ant_forest) {
    //     click_widget(home);
    //     click(100, 2180);   //首页坐标，防止点击控件失败，重复点击，不影响
    // }
    //
    // ant_forest = antIcon.findOne(timeout);
    // click_widget(ant_forest);
    // sleep(timeout * 5);
    // let hz = hezhong.findOne(timeout);
    // if (!hz) {
    //     sleep(timeout * 20);
    // }
    // ant_forest = antIcon.findOne(timeout);
    // if (!hz && ant_forest) {
    //     click(945, 530);    //蚂蚁森林坐标，防止点击控件失败
    //     sleep(timeout * 5);
    // }

    // hz = hezhong.findOne(timeout);
    // if (!hz) {
    //     sleep(timeout * 10);
    // }
    // hz = hezhong.findOne(timeout);
    // if (hz) {
    //     log("进入蚂蚁森林成功");
    // } else {
    //     log("进入蚂蚁森林失败");
    //     let time = new Date();
    //     images.captureScreen('/sdcard/脚本/' + time.getMinutes() + '_' + time.getSeconds() + '.jpg');
    //     return false;
    // }
    //
    // return true;
}


function unique(arr) {
    let hash = {};
    let result = [];
    for (let i = 0, len = arr.length; i < len; i++) {
        if (!hash[arr[i]]) {
            result.push(arr[i]);
            hash[arr[i]] = true;
        }
    }
    return result
}

function get_my_energy() {
    click(138, 648);
    sleep(2000);
    log("get_my_energy");
    let selectors = energy.find();
    if (!selectors.empty()) {
        let points = selectors.map(e => [e.bounds().centerX(), e.bounds().centerY() - 60]);
        // log(unique(points));
        unique(points).map(e => click(e[0], e[1]));
        sleep(timeout)
    }
}

function get_energy() {
    let handSpace = images.findImage(images.capetureScreen(), handimg, {region: [1000, 200]});
    while (handSpace) {
        //log(W / 2, handSpace.y + 20)
        sleep(timeout);
        click(W / 2, handSpace.y + 20);// 可能按不进去
        sleep(2000);
        if (inforest.findOne(timeout)) {  //可能缓冲很久
            let selectors = energy.find();
            if (!selectors.empty()) {
                let points = selectors.map(e => [e.bounds().centerX(), e.bounds().centerY() - 60]);
                // log(unique(points));
                unique(points).map(e => click(e[0], e[1]));
                sleep(timeout)
            }
            back();
            sleep(1000)
        }
        handSpace = images.findImage(images.captureScreen(), handimg, {region: [1000, 200]});
    }
}

function get_friends_energy() {
    //moreFriend.waitFor();
    let mf = moreFriend.findOne(timeout);
    if (mf) {
        mf.click();
    }
    else {
        swipe_count = 3;
        for (let i = 0; i < swipe_count; i++) {
            swipe(100, 1900, 100, 400, 100);
            sleep(timeout);
        }
        click(500, 800);
    }
    sleep(3000);
    //noMore控件初始找不到，后来能找到但滑不到最下面，往下滑十次
    swipe_count = 10;
    for (let i = 0; i < swipe_count; i++) {
        get_energy();
        swipe(100, 1900, 100, 400, 100);
        sleep(timeout);
    }

    back();
    sleep(timeout * 3);
}

function unlock_screen() {
    max_retry_times = 3;
    let screen_on = device.isScreenOn();
    while (!screen_on && max_retry_times) {
        device.wakeUp();
        sleep(timeout);
        swipe(200, 1600, 900, 400, 300);
        //swipe(100, 1900, 100, 400, 100);
        sleep(timeout);
        let pass = [[540, 1480], [220, 1260], [860, 1480], [860, 1260], [220, 1260], [540, 1920]];
        pass.map(e => click(e[0], e[1]));

        sleep(timeout * 12);
        screen_on = device.isScreenOn();
        max_retry_times -= 1;
    }

    if (max_retry_times) {
        log("解锁成功");
        images.captureScreen('/sdcard/脚本/解锁后截图.jpg');
    } else {
        log("解锁失败");
        let time = new Date();
        images.captureScreen('/sdcard/脚本/' + time.getMinutes() + '_' + time.getSeconds() + '.jpg');
        return false;
    }
    return true;
}

if (unlock_screen()) {
    let is_forest = enter_ant_forest();
    while (1) {
        get_my_energy();
        get_friends_energy();

        let time = new Date();
        if (time.getHours() > 7) {
            break
        }
        if (time.getMinutes() < 30) {
            sleep(timeout * 60);
        } else {
            sleep(timeout * 60 * 5)
        }
        //is_forest = enter_ant_forest();
    }
    back();
}
