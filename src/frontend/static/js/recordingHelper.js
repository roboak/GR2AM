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

        $(function () {
            $('#tooltipDiv[data-toggle="tooltip"]').attr('title', "Record selected gesture");

            new bootstrap.Tooltip($('#tooltipDiv[data-toggle="tooltip"]')[0])
        })
    }
}, false);


function clickRecord() {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/recordClick", true);
    // xhr.setRequestHeader("Content-type", "frontend/x-www-form-urlencoded");
    xhr.send();
}


function clickNext() {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/nextClick", true);
    // xhr.setRequestHeader("Content-type", "frontend/x-www-form-urlencoded");
    xhr.send();

    i = parseInt($(".progress-bar").text().split('%')[0])

    i += 10;
    // update progress bar
    $(".progress-bar").css("width", i + "%").text(i + " %");
}

function clickRedo() {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/redoClick", true);
    // xhr.setRequestHeader("Content-type", "frontend/x-www-form-urlencoded");
    xhr.send();
}

function removeCapture(gesture_id) {
    var xhr = new XMLHttpRequest()

    xhr.onreadystatechange = function () {
        if (this.readyState === this.DONE)
            var xhr2 = new XMLHttpRequest()
        xhr2.open("GET", "/")
        xhr2.onreadystatechange = function () {
            if (this.readyState === this.DONE)
                location.reload()
        }
        xhr2.send()
    }

    xhr.open("GET", "/remove-gesture/" + gesture_id)
    xhr.send()
}

function callModelGeneration() {
    $('.spinner-border').show();

    var xhr = new XMLHttpRequest();
    xhr.open("GET", "{{url_for('record_gesture.generate_model')}}", true);
    xhr.onload = function (e) {
        if (xhr.readyState === 4) {
            $('.spinner-border').hide();
            imgM = document.getElementById("modalImg")
            imgM.src = "/matrix"
            $('#matrixModal').modal('show');
        }
    };
    xhr.send();
}

function checkProgress() {
    $("input[type='radio'][name='selectGesture']").each(function () {

        let g_name = $(this).val()
        var xhr = new XMLHttpRequest();

        xhr.open("GET", "/progress/" + g_name);
        xhr.onload = function (e) {
            if (xhr.readyState === 4)
                if (xhr.status === 200) {
                    let val = xhr.responseText;
                    if (val !== "0") {
                        //Color text red
                        $("label[for='radio_" + g_name + "']").css("color", "red")
                        let btn = document.getElementById("btn_gen_model")
                        btn.disabled = true;
                    }
                }
        };
        xhr.send();
    });
}
