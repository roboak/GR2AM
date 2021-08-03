
var form = document.getElementById('gestures');
form.addEventListener('change', function (e) {
    /* Determine if the e.target (radio that's clicked) is NOT e.currentTarget (#roles) */
    if (e.target !== e.currentTarget) {
        // Reference the button
        var btn = document.getElementById("recordStart");
        var btn2 = document.getElementById("mapBtn");

        // Enable button
        if (btn != null) btn.disabled = false;
        if (btn2 != null) btn2.disabled = false;
    }
}, false);


function clickRecord() {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/recordClick", true);
    // xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhr.send();
}

var i = 0;

function clickNext() {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/nextClick", true);
    // xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhr.send();

    i += 10;
    // update progress bar
    $(".progress-bar").css("width", i + "%").text(i + " %");
}

function clickRedo() {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/redoClick", true);
    // xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhr.send();
}

function removeCapture(gesture_id){
    var xhr = new XMLHttpRequest()

    xhr.onreadystatechange = function () {
        if (this.readyState === this.DONE)
            location.reload()
    }

    xhr.open("GET", "/remove-gesture/" + gesture_id)
    xhr.send()
}
