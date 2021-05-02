function displayModal(title, message) {
    document.getElementById("modal-title").innerHTML = title
    document.getElementById("modal-body").innerHTML = message
    $('#resultModal').modal('show')
}

function fileValidation() {
    var inputFile =
        document.getElementById('file');
    var fileName = inputFile.value;
    console.log(fileName)
    var allowedExtension = /\.m4a$/i;
    if (!allowedExtension.exec(fileName)) {
        console.log("here")
        displayModal('Upload failed', 'Please upload MP4 file')
        inputFile.value = '';
        document.getElementById("fileUpload").innerHTML = 'Upload Audio'
        return false
    } else {
        document.getElementById("fileUpload").innerHTML = fileName.split('\\').pop();
        return true
    }
}

function fileValidation2() {
    var inputFile =
        document.getElementById('file2');
    var fileName = inputFile.value;
    console.log(fileName)
    var allowedExtension = /\.*$/i;
    if (!allowedExtension.exec(fileName)) {
        console.log("here")
        displayModal('Upload failed', 'Please upload txt file')
        inputFile.value = '';
        document.getElementById("fileUpload").innerHTML = 'Upload Audio'
        return false
    } else {
        document.getElementById("fileUpload2").innerHTML = fileName.split('\\').pop();
        return true
    }
}

// var form = document.getElementById("form2")
// form.addEventListener('submit', function (f) {
//     f.preventDefault()
//     var transcript = document.getElementById("transcripts")
//     console.log(transcript.value)
//     fetch(window.location.origin + "/transcript", {
//             method: "POST",
//         headers : {
//         'Content-Type': 'application/json',
//         'Accept': 'application/json'
//        },
//             body: JSON.stringify({
//                     transcript: transcript.value
//                 })
//         })
//
//         .then(function (response) {
//                 console.log(response.status)
//             })
// })

URL = window.URL || window.webkitURL;

var gumStream;                      //stream from getUserMedia()
var rec;                            //Recorder.js object
var input;                          //MediaStreamAudioSourceNode we'll be recording

// shim for AudioContext when it's not avb.
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext //audio context to help us record

var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");
var pauseButton = document.getElementById("pauseButton");

//add events to those 2 buttons
recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
pauseButton.addEventListener("click", pauseRecording);

function startRecording() {
    console.log("recordButton clicked");

    /*
        Simple constraints object, for more advanced audio features see
        https://addpipe.com/blog/audio-constraints-getusermedia/
    */

    var constraints = { audio: true, video:false }

    /*
        Disable the record button until we get a success or fail from getUserMedia()
    */

    recordButton.disabled = true;
    stopButton.disabled = false;
    pauseButton.disabled = false

    /*
        We're using the standard promise based getUserMedia()
        https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
    */

    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");

        /*
            create an audio context after getUserMedia is called
            sampleRate might change after getUserMedia is called, like it does on macOS when recording through AirPods
            the sampleRate defaults to the one set in your OS for your playback device

        */
        audioContext = new AudioContext();

        //update the format
        document.getElementById("formats").innerHTML="Format: 1 channel pcm @ "+audioContext.sampleRate/1000+"kHz"

        /*  assign to gumStream for later use  */
        gumStream = stream;

        /* use the stream */
        input = audioContext.createMediaStreamSource(stream);

        /*
            Create the Recorder object and configure to record mono sound (1 channel)
            Recording 2 channels  will double the file size
        */
        rec = new Recorder(input,{numChannels:1})

        //start the recording process
        rec.record()

        console.log("Recording started");

    }).catch(function(err) {
        //enable the record button if getUserMedia() fails
        recordButton.disabled = false;
        stopButton.disabled = true;
        pauseButton.disabled = true
    });
}

function pauseRecording(){
    console.log("pauseButton clicked rec.recording=",rec.recording );
    if (rec.recording){
        //pause
        rec.stop();
        pauseButton.innerHTML="Resume";
    }else{
        //resume
        rec.record()
        pauseButton.innerHTML="Pause";

    }
}

function stopRecording() {
    console.log("stopButton clicked");

    //disable the stop button, enable the record too allow for new recordings
    stopButton.disabled = true;
    recordButton.disabled = false;
    pauseButton.disabled = true;

    //reset button just in case the recording is stopped while paused
    pauseButton.innerHTML="Pause";

    //tell the recorder to stop the recording
    rec.stop();

    //stop microphone access
    gumStream.getAudioTracks()[0].stop();

    //create the wav blob and pass it on to createDownloadLink
    rec.exportWAV(createDownloadLink);
}

function createDownloadLink(blob) {
    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    au.controls = true;
    au.src = url;
    var li = document.createElement('li');
    li.appendChild(au);
    var li = document.createElement('li');
    var filename = new Date().toISOString();
    //upload link
    var upload = document.createElement('a');
    upload.href="#";
    upload.innerHTML = "Upload";
    upload.addEventListener("click", function(event){
          var xhr=new XMLHttpRequest();
          xhr.onload=function(e) {
              if(this.readyState === 4) {
                  console.log("Server returned: ",e.target.responseText);
              }
          };
          var fd=new FormData();

          fd.append("audio_data",blob, filename);
          xhr.open("POST","/rec?filename="+filename,true);
          xhr.send(fd);
    })
    li.appendChild(document.createTextNode (" "))//add a space in between
    li.appendChild(upload)//add the upload link to li

    //add the li element to the ol
    recordingsList.appendChild(li);
}
let is_completed = false
out = document.getElementById("output")
function populateOutput() {
fetch(window.location.origin + "/output")
                .then(function (resp) {
                    console.log(resp)
                    return resp.json()
                }) .then(function(json) {
                    if(!json.hasOwnProperty("none")) {

                        let visited = new Map()
                        is_completed = true
                        while (out.firstChild) {
                            out.removeChild(out.firstChild)
                        }
                    for (var key in json) {
                        if (json.hasOwnProperty(key)) {
                            if(!visited.has(key.split("/")[0])) {
                                let heading = document.createElement("div")
                                heading.classList.add("h3")
                                heading.innerText = key.split("/")[0]
                                out.appendChild(heading)
                                let row = document.createElement("div")
                                row.classList.add("row")
                                row.id = key.split("/")[0]
                                visited.set(key.split("/")[0])
                                out.appendChild(row)
                            }
                            let row = document.getElementById(key.split("/")[0])

                            let col1 = document.createElement("div")
                            col1.classList.add("col-md-4")
                            let img1 = document.createElement("img")
                            img1.classList.add("img")
                            img1.src =json[key]
                            img1.alt = ""
                            col1.appendChild(img1)
                            row.appendChild(col1)
                        }
                    }
                    }
})
}

let timer = setInterval(function () {
            populateOutput()
    console.log("here")
            if (is_completed == true) {
                clearInterval(timer)
            }
        }, 5000);
