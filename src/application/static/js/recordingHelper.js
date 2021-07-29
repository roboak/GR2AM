// Reference #roles
var form = document.getElementById('gestures');

/* Register the change element to #roles
|| When clicked...
*/
form.addEventListener('change', function (e) {

    /* Determine if the e.target (radio that's clicked)
    || is NOT e.currentTarget (#roles)
    */
    if (e.target !== e.currentTarget) {

        // Assign variable to e.target
        var target = e.target;

        // Find the textNode next to target
        var label = target.nextSibling;

        // Reference the #display
        var display = document.getElementById('display');

        // Display the <label>s text and radio value
        display.value = label.textContent + ' - Rank: ' + target.value;

        // Reference the submit button
        var btn = document.querySelector('[type=submit]');

        // Enable submit button
        btn.disabled = false;

        // call rolrDist() passing the target,value
        roleDist(target.value);
    }
}, false);

function roleDist(rank) {

    switch (rank) {
        case '4':
            alert('Rank 4 - Limited Access');
            // Take user to landing page
            break;

        case '3':
            alert('Rank 3 - Basic Access');
            // Take user to dashboard
            break;

        case '2':
            alert('Rank 2 - Advanced Access');
            // Take user to database
            break;

        case '1':
            alert('Rank 1 - Full Access');
            // Take user to admin panel
            break;
    }
}


function clickRecord() {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/recordClick", true);
    // xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhr.send();
}

$( window ).on( "load", function() {
    i = 0;
});

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
