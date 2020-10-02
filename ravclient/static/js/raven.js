(function () {

    // var net = require('net-browserify');
    //
    // var client = new net.Socket();
    // client.connect(65433, '127.0.0.1', function() {
    //     console.log('Connected');
    //     client.write('Hello, server! Love, Client.');
    // });
    //
    // client.on('data', function(data) {
    //     console.log('Received: ' + data);
    //     client.destroy(); // kill client after server's response
    // });

    var socket = io('ws://localhost:9999/ravjs', {
        query: {
              "client_name": "ravjs"
        }});

    socket.on('op', function(d){
        console.log(d);
        var data = JSON.parse(d);
        console.log(data);

        //Acknowledge op
        socket.emit("acknowledge", JSON.stringify({
            "op_id": data.op_id,
            "message": "Op received"
        }));

        // Perform
        let operation_type = data ["op_type"];
        let operator = data ["operator"];
        if(operation_type && operator) {
            compute_operation(data);
        }
    });

    socket.on('connect', function(d){
        console.log("connected");

        socket.emit("ask_op", JSON.stringify({
            "message": "Send me an aop"
        }))
    });

    socket.on("ping", function(message) {
        console.log(message);
        console.log("Received PING");

        console.log("Sending PONG");
        socket.emit("pong", JSON.stringify({
            "message": "PONG"
        }));
    });

    /* Socket connection and operations */

    // var webSocket = new WebSocket('ws://127.0.0.1:65433');
    //
    // webSocket.onmessage = function (e) {
    //     var data = JSON.parse(e.data);
    //     console.log(data);
    //     let operation_type = data ["op_type"];
    //     let operator = data ["operator"];
    //     if(operation_type && operator) {
    //         compute_operation(data);
    //     }
    // };
    //
    // webSocket.onopen = function (e) {
    //     webSocket.send(JSON.stringify({
    //         "client_id":guidGenerator()
    //     }));
    // };
    //
    // webSocket.onclose = function (e) {
    //     console.error('Web socket closed unexpectedly');
    // };
    //

    function compute_operation(payload) {
        switch(payload.operator) {
            case "matrix_multiplication":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.matMul(y);
                    console.log("Computing matrix multiplication");
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));                    
                }
                catch(error) {
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "failure"
                    }));
                }       
                break;

            case "negation":
                try {
                    x = tf.tensor(payload.values[0]);
                    console.log("Computing negation");
                    result = x.neg();
                    console.log("Result:"+result);
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "failure"
                    }));
                }
                break;

            case "addition":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.add(y);
                    console.log("Computing addition");
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "failure"
                    }));
                }
                break;

            case "division":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.div(y);
                    console.log("Computing division");
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "failure"
                    }));
                }
                break;

            case "exponential":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.exp();
                    console.log("Computing exponential");
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "failure"
                    }));
                }
                break;

            case "transpose":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.transpose();
                    console.log("Computing transpose");
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "failure"
                    }));
                }
                break;

            case "natural_log":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.log();
                    console.log("Computing natural log");
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "failure"
                    }));
                }
                break;

            case "element_wise_multiplication":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.mul(y);
                    console.log("Computing element wise multiplication");
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "failure"
                    }));

                }
                break;

            case "subtraction":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.sub(y);
                    console.log("Computing subtraction");
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "failure"
                    }));

                }
                break;
            case "linear":
                try {
                    console.log("Computing linear");
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': payload.values[0],
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "failure"
                    }));
                }
                break;

            case "matrix_sum":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.sum()
                    console.log("Computing matrix_sum");
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("result", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "failure"
                    }));
                }
                break;
        }
    }

    $(window).bind('beforeunload', function(){
        socket.disconnect();
        return 'Are you sure you want to leave?';
    });

    // function guidGenerator() {
    //     let S4 = function () {
    //         return (((1 + Math.random()) * 0x10000) | 0).toString(16).substring(1);
    //     };
    //     return (S4() + S4() + "-" + S4() + "-" + S4() + "-" + S4() + "-" + S4() + S4() + S4());
    // }

})();

