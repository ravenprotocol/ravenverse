(function () {
    var socket_server_url = 'ws://'+window.location.hostname+':9999/ravjs';

    var socket = io(socket_server_url, {
        query: {
              "client_name": "ravjs"
        }});

    socket.on('op', function(d){
        $(".clientStatus").append("Op received");

        var data = JSON.parse(d);

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
        console.log("Connected");

        $(".clientStatus").html("Connected");

        socket.emit("get_op", JSON.stringify({
            "message": "Send me an aop"
        }))
    });

    socket.on('disconnect', function(d) {
        console.log("Disconnected");
         $(".clientStatus").html("Disconnected");
    });

    socket.on("ping", function(message) {
        console.log(message);
        console.log("Received PING");

        console.log("Sending PONG");
        socket.emit("pong", JSON.stringify({
            "message": "PONG"
        }));
    });

    function compute_operation(payload) {
        switch(payload.operator) {
            case "matrix_multiplication":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.matMul(y);
                    console.log("Computing matrix multiplication");
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));                    
                }
                catch(error) {
                    socket.emit("op_completed", JSON.stringify({
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
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("op_completed", JSON.stringify({
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
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("op_completed", JSON.stringify({
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
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("op_completed", JSON.stringify({
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
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("op_completed", JSON.stringify({
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
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("op_completed", JSON.stringify({
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
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("op_completed", JSON.stringify({
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
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("op_completed", JSON.stringify({
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
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("op_completed", JSON.stringify({
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
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': payload.values[0],
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("op_completed", JSON.stringify({
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
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values':payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                }
                catch(error) {
                    socket.emit("op_completed", JSON.stringify({
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
        $(".clientStatus").html("Disconnected");
        socket.disconnect();
        return 'Are you sure you want to leave?';
    });

})();

