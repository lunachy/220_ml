// auto();
// console.show();
//搜狗输入法坐标
// Y/X   300 540 780
// 1675   1   2   3
// 1845   4   5   6
// 2020   7   8   9
// 2190      空格
// let pass = [[540, 1675], [220, 1455], [860, 1675], [860, 1455], [220, 1455], [540, 2100]];
// pass.map(e => click(e[0], e[1]));

let x = [300, 540, 780];
let y = [1675, 1845, 2020, 2190];
let a0 = [x[1], y[3]];
let a1 = [x[0], y[0]];
let a2 = [x[1], y[0]];
let a3 = [x[2], y[0]];
let a4 = [x[0], y[1]];
let a5 = [x[1], y[1]];
let a6 = [x[2], y[1]];
let a7 = [x[0], y[2]];
let a8 = [x[1], y[2]];
let a9 = [x[2], y[2]];

function comment() {
    click(965, 1410);  //进入评论界面
    sleep(1000);
    click(500, 2206);   //点击输入框
    sleep(1000);
    click(988, 1668);   //输入法x
    sleep(1000);

    // click(100, 1520);  //粘贴
    click(638, 1525);
    sleep(1000);
    click(880, 1800);
    sleep(1000);

    click(980, 1400);   //点击发送
    sleep(1000);
    back();
    sleep(1000);
}

images.requestScreenCapture();
launch("com.waqu.android.firebull");
sleep(5000);
click(750, 1280);
sleep(500);
swipe(500, 1900, 500, 300, 100);
setClip("互赞互粉");

const zanimg = images.read('/sdcard/脚本/zan.jpg');
while (1) {
    // let ret = images.findImage(images.captureScreen(), zanimg, {region: [850, 1000, 130, 200]});
    // if (ret == null) {
    //     sleep(1000 * 60);
    //     continue;
    // }
    sleep(1000);
    click(965, 1070);  //关注
    sleep(1000);
    click(965, 1200);  //点赞
    sleep(1000);
    comment();
    sleep(8000);
    swipe(500, 1900, 500, 300, 100);
    sleep(1000);
}
