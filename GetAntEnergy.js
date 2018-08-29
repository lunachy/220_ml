// auto();
// console.show()
// Y/X   220 540 860
// 1455   1   2   3
// 1675   4   5   6
// 1885   7   8   9
// 2100       0
const handimg = images.read('/sdcard/脚本/take.png');
//const gimg = images.read('/sdcard/脚本/g.jpg');
var [W, H] = [device.width, device.height];

const alipayHome = className("android.widget.TextView").text("首页");
const antIcon = className("android.widget.TextView").text("蚂蚁森林");
const hezhong = className("android.widget.Button").desc("合种");
const myEnergyCat = className("android.view.View").descMatches(/线下支付|行走|生活缴费/);
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
var max_retry_times = 5;
images.requestScreenCapture();

function enter_ant_forest() {
    max_retry_times = 5;
    launch(alipay_package); //启动支付宝
    sleep(timeout);
    var home = alipayHome.findOne(timeout);
    while (!home && max_retry_times) {
        log("进入支付宝首页失败，可能在支付宝别的界面，按返回键直到首页");
        back();
        max_retry_times -= 1;
        sleep(timeout);
        home = alipayHome.findOne(timeout);
    }
    if (max_retry_times) {
        log("进入支付宝首页成功");
    } else {
        log("进入支付宝首页失败");
        var time = new Date();
        images.captureScreen('/sdcard/脚本/' + time.getMinutes() + '_' + time.getSeconds() + '.jpg');
        return false;
    }

    click_widget(home);
    max_retry_times = 3;
    var ant_forest = antIcon.findOne(timeout);
    while (!ant_forest && max_retry_times) {
        click_widget(home);
        max_retry_times -= 1;
        sleep(timeout);
        ant_forest = antIcon.findOne(timeout);
    }
    if (max_retry_times) {
        log("进入支付宝首页成功1");
    } else {
        log("进入支付宝首页失败1");
        var time = new Date();
        images.captureScreen('/sdcard/脚本/' + time.getMinutes() + '_' + time.getSeconds() + '.jpg');
        return false;
    }

    click_widget(ant_forest);
    max_retry_times = 3;
    var hz = hezhong.findOne(timeout);
    while (!hz && max_retry_times) {
        click_widget(ant_forest);
        max_retry_times -= 1;
        sleep(timeout * 10);
        hz = hezhong.findOne(timeout);
    }
    if (max_retry_times) {
        log("进入蚂蚁森林成功");
    } else {
        log("进入支蚂蚁森林失败");
        var time = new Date();
        images.captureScreen('/sdcard/脚本/' + time.getMinutes() + '_' + time.getSeconds() + '.jpg');
        return false;
    }

    return true;
}


function unique(arr) {
    var hash = {};
    var result = [];
    for (var i = 0, len = arr.length; i < len; i++) {
        if (!hash[arr[i]]) {
            result.push(arr[i]);
            hash[arr[i]] = true;
        }
    }
    return result
}


function get_my_energy() {
    log("get_my_energy");
    hezhong.waitFor();
    var points = className("android.widget.Button").filter(function (o) {
        var desc = o.contentDescription;
        return (null !== desc.match(/^收集能量|^$/));
    }).find().map(e => [e.bounds().centerX(), e.bounds().centerY()]);
    if (points.length){
        sleep(timeout * 8);
        points.map(e => click(e[0], e[1]));
    }
    sleep(timeout);
    hezhong.waitFor();
}

function findHand() {
    try {
        return images.findImage(images.captureScreen(), handimg, {
            region: [1000, 200]
        })
    } catch (err) {
        log(err);
        return null;
    }
}

function get_energy() {
    var handSpace = findHand();
    while (handSpace) {
        //log(W / 2, handSpace.y + 20)
        sleep(timeout);
        click(W / 2, handSpace.y + 20);// 可能按不进去
        sleep(2000);
        if (inforest.findOne(timeout)) {  //可能缓冲很久
            var selectors = canget.find();
            if (!selectors.empty()) {
                var points = selectors.map(e => [e.bounds().centerX(), e.bounds().centerY() - 100]);
                // log(unique(points))
                unique(points).map(e => click(e[0], e[1]));
                sleep(timeout)
            }
            back();
            sleep(1000)
        }
        handSpace = findHand();
    }
}

function get_friends_energy() {
    moreFriend.waitFor();
    var mf = moreFriend.findOne(timeout);
    mf.click();
    sleep(3000);
    friendRank.waitFor();
    //noMore控件初始找不到，后来能找到但滑不到最下面，往下滑十次
    swipe_count = 10;
    for (let i = 0; i < swipe_count; i++) {
        get_energy();
        swipe(100, 1900, 100, 400, 100);
        sleep(timeout);
    }

    back();
    sleep(timeout);
    back();
    sleep(timeout);
}

function unlock_screen() {
    max_retry_times = 3;
    var screen_on = device.isScreenOn();
    while (!screen_on && max_retry_times) {
        device.wakeUp();
        sleep(timeout);
        swipe(100, 1900, 100, 400, 100);
        sleep(timeout);
        var pass = [[540, 1675], [220, 1455], [860, 1675], [860, 1455], [220, 1455], [540, 2100]];
        pass.map(e => click(e[0], e[1]));

        sleep(timeout * 12);
        screen_on = device.isScreenOn();
        max_retry_times -= 1;
    }

    if (max_retry_times) {
        log("解锁成功");
    } else {
        log("解锁失败");
        var time = new Date();
        images.captureScreen('/sdcard/脚本/' + time.getMinutes() + '_' + time.getSeconds() + '.jpg');
        return false;
    }
    return true;
}

if (unlock_screen()) {
    var is_forest = enter_ant_forest();
    while (is_forest) {
        get_my_energy();
        get_friends_energy();
 
        var time = new Date();
        if (time.getHours() > 7) {
            break
        }
        if (time.getMinutes() < 30) {
            sleep(timeout * 60);
        } else {
            sleep(timeout * 60 * 5)
        }
        is_forest = enter_ant_forest();
    }
}
