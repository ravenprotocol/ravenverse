(function () {
    var socket_server_url = 'ws://' + window.location.hostname + ':9999/ravjs';

    var socket = io(socket_server_url, {
        query: {
            "client_name": "ravjs"
        }
    });

    socket.on('op', function (d) {
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
        if (operation_type && operator) {
            compute(data);
        }
    });

    socket.on('connect', function (d) {
        console.log("Connected");

        $(".clientStatus").html("Connected");

        socket.emit("get_op", JSON.stringify({
            "message": "Send me an aop"
        }))
    });

    socket.on('disconnect', function (d) {
        console.log("Disconnected");
        $(".clientStatus").html("Disconnected");
    });

    socket.on("ping", function (message) {
        console.log(message);
        console.log("Received PING");

        console.log("Sending PONG");
        socket.emit("pong", JSON.stringify({
            "message": "PONG"
        }));
    });

    function compute(payload) {
        console.log("Computing "+payload.operator);
        switch (payload.operator) {
            case "linear":
                try {
                    emit_result(payload, payload.values[0]);
                } catch (e) {
                    emit_error(payload);
                }
                break;
            case "addition":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.add(y).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "subtraction":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.sub(y).arraySync();
                    emit_result(payload, result)
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "multiplication":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.multiply(y).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    result = error.message;
                    emit_error(payload, result);
                }
                break;
            case "division":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.div(y).arraySync();
                    emit_result(payload, result)
                } catch (error) {
                     emit_error(payload, error);
                }
                break;
            case "negation":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.neg().arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error)
                }
                break;
            case "exponential":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.exp().arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "natural_log":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.log().arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "power":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.pow(y).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "square":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.pow(2.0).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "cube":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.pow(3.0).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "square_root":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.sqrt().arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "cube_root":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.pow(1.0/3.0).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "absolute":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.abs().arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "matrix_multiplication":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.matMul(y).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "multiply":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.mul(y).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "dot":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.dot(y).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "transpose":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.transpose().arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "matrix_sum":
                try {
                    x = tf.tensor(payload.values[0]);
                    let params = payload.params;
                    if('axis' in params){
                        let axis = params.axis;
                        result = x.sum(axis).arraySync();
                    }else{
                        result = x.sum().arraySync();
                    }
                    emit_result(payload, result)
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "sort":
                emit_error(payload, {message: "This method is not implemented yet"});
                break;
            case "split":
                emit_error(payload, {message: "This method is not implemented yet"});
                break;
            case "reshape":
                emit_error(payload, {message: "This method is not implemented yet"});
                break;
            case "concatenate":
                emit_error(payload, {message: "This method is not implemented yet"});
                break;
            case "min":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.min().arraySync();
                    emit_result(payload, result)
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "max":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.max().arraySync();
                    emit_result(payload, result)
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "unique":
                emit_error(payload, {message: "This method is not implemented yet"});
                break;
            case "argmax":
                try {
                    x = tf.tensor(payload.values[0]);
                    let params = payload.params;
                    if('axis' in params){
                        let axis = params.axis;
                        result = x.argMax(axis).arraySync();
                    }else{
                        result = x.argMax().arraySync();
                    }
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "greater":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.greater(y).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "greater_equal":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.greaterEqual(y).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "less":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.less(y).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "less_equal":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.lessEqual(y).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "equal":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.equal(y).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "not_equal":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.notEqual(y).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "logical_and":
                try {
                    x = tf.tensor(payload.values[0], undefined, 'bool');
                    y = tf.tensor(payload.values[1], undefined, 'bool');
                    result = x.logicalAnd(y).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "logical_or":
                try {
                    x = tf.tensor(payload.values[0], undefined, 'bool');
                    y = tf.tensor(payload.values[1], undefined, 'bool');
                    result = x.logicalOr(y).arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "logical_not":
                try {
                    x = tf.tensor(payload.values[0], undefined, 'bool');
                    result = x.logicalNot().arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "logical_xor":
                try {
                    x = tf.tensor(payload.values[0], undefined, 'bool');
                    y = tf.tensor(payload.values[1], undefined, 'bool');
                    result = x.logicalXor().arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "mean":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.mean().arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
            case "average":
                try {
                    x = tf.tensor(payload.values[0]);
                    result = x.mean().arraySync();
                    emit_result(payload, result);
                } catch (error) {
                    emit_error(payload, error);
                }
                break;
        }
    }

    function emit_result(payload, result) {
        console.log("Emit Success");
        console.log(payload);
        console.log(result);

        socket.emit("op_completed", JSON.stringify({
            'op_type': payload.op_type,
            'result': result,
            'values': payload.values,
            'operator': payload.operator,
            "op_id": payload.op_id,
            "status": "success"
        }));
    }

    function emit_error(payload, error) {
        console.log("Emit Error");
        console.log(payload);
        console.log(error);

        socket.emit("op_completed", JSON.stringify({
            'op_type': payload.op_type,
            'result': error.message,
            'values': payload.values,
            'operator': payload.operator,
            "op_id": payload.op_id,
            "status": "failure"
        }));
    }

    function compute_operation(payload) {
        switch (payload.operator) {

            // Arithmetic
            case "linear":
                try {
                    console.log("Computing linear");
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': payload.values[0],
                        'values': payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                } catch (error) {
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values': payload.values,
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
                        'values': payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                } catch (error) {
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values': payload.values,
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
                        'values': payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                } catch (error) {
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values': payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "failure"
                    }));

                }
                break;

            case "multiplication":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.multiply(y);
                    console.log("Computing " + payload.operator);
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values': payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                } catch (error) {
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values': payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "failure"
                    }));

                }
                break;

            case "matrix_multiplication":
                try {
                    x = tf.tensor(payload.values[0]);
                    y = tf.tensor(payload.values[1]);
                    result = x.matMul(y);
                    console.log("Computing matrix multiplication");
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values': payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                } catch (error) {
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values': payload.values,
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
                    console.log("Result:" + result);
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': result.arraySync(),
                        'values': payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                } catch (error) {
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values': payload.values,
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
                        'values': payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                } catch (error) {
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values': payload.values,
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
                        'values': payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                } catch (error) {
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values': payload.values,
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
                        'values': payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                } catch (error) {
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values': payload.values,
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
                        'values': payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                } catch (error) {
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values': payload.values,
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
                        'values': payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                } catch (error) {
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values': payload.values,
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
                        'values': payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "success"
                    }));
                } catch (error) {
                    socket.emit("op_completed", JSON.stringify({
                        'op_type': payload.op_type,
                        'result': error.message,
                        'values': payload.values,
                        'operator': payload.operator,
                        "op_id": payload.op_id,
                        "status": "failure"
                    }));
                }
                break;
        }
    }

    $(window).bind('beforeunload', function () {
        $(".clientStatus").html("Disconnected");
        socket.disconnect();
        return 'Are you sure you want to leave?';
    });

})();

