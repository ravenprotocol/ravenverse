$(document).ready(function () {
	$("#core").addClass('active');
	setDisplayNone();
	$("#coreContainer").css("display", "flex");

	$("input[id=chooseFile]").change(function() {
		$(".uploadFileLabel").text($(this).val());
	});

	$('input[type=radio][value=boston_house_prices]').prop("checked", true);
	$("#chooseFileLinear").prop("disabled", true);
	$("#fileUploadBtnLinear").prop("disabled", true);


});

function removeClasses() {
	$("#core").removeClass('active');
	$("#linear").removeClass('active');
	$("#logistic").removeClass('active');
}

function setDisplayNone() {
	$("#coreContainer").css("display", "none");
	$("#linearContainer").css("display", "none");
	$("#logisticContainer").css("display", "none");
}

$("#core").click(function () {
	removeClasses();
	setDisplayNone();
	$("#core").addClass('active');
	$("#coreContainer").css("display", "flex");
});

$("#linear").click(function () {
	removeClasses();
	setDisplayNone();
	$("#linear").addClass('active');
	$("#linearContainer").css("display", "flex");
	$("#datasetLinear").addClass('active');
	$("#inputLinear").removeClass('active');
	setDatasetLinear();
});

$("#logistic").click(function () {
	removeClasses();
	setDisplayNone();
	$("#logistic").addClass('active');
	$("#logisticContainer").css("display", "flex");
	$("#datasetLogistic").addClass('active');
	$("#inputLogistic").removeClass('active');
});

$("select[name='coreDropdown']").change(function() {
	let operatorNumber = getOperationNumber($(this).val());
	if(operatorNumber >= 7) {
		$("#input2").prop("disabled", true);
		// $("#input2Dropdown").prop("disabled", true);
	}
	else {
		$("#input2").prop("disabled", false);
		// $("#input2Dropdown").prop("disabled", false);
	}

	$("#errorBox").remove();
})

function pollResult(opId, timer) {
	$.ajax({
		url: `http://127.0.0.1:8000/app/result/${opId}`,
		type: "GET",
		success: function(res) {
			if(res.status == "computed") {
				let newElement = `<div id="calcuateCoreResult"></div>`;
				$("#calcuateCoreContainer").append(newElement);
				$("#calcuateCoreResult").text(`Result: ${JSON.stringify(res.result)}`);
				$("#calcuateCoreBtn").text("Calculate");
				clearInterval(timer);
			}
		}
	});
}

function getOperationNumber(operation) {
	let obj = {
		"Addition": 1,
		"Subtraction": 2,
		"Matrix Multiplication": 3,
		"Multiplication": 4,
		"Division": 5,
		"Element-wise Multiplication": 6,
		"Negation": 7,
		"Exponential": 8,
		"Transpose": 9,
		"Natural Log": 10,
		"Matrix Sum": 11,
		"Linear": 12
	}

	return obj[operation];
}

function createRequestObj() {
	let obj = {
		operation: parseInt($("select[name='coreDropdown']").val()),
		data1: $("#input1").val(),
	};

	let operatorNumber = getOperationNumber($("select[name='coreDropdown']").val());

	if(operatorNumber >= 7) {
		obj["data2"] = null;
		
	}
	else {
		obj["data2"] = $("#input2").val();

		// if($("select[name='input1Dropdown']").val() === "matrix" && $("select[name='input2Dropdown']").val() === "matrix" && obj.operation == 4) {
		// 	obj.operation = 1;
		// }
	}

	return obj;
}

function checkInputs() {
	let check = "";
	let operatorNumber = getOperationNumber($("select[name='coreDropdown']").val());
	console.log(operatorNumber);
	
	if(($("select[name='coreDropdown']").val() == "select")) {
		check = "select";
	}
	if(($("#input1").val() == "")) {
		check = "input1";
	}
	if(($("#input2").val() == "") && operatorNumber < 7) {
		check = "input2";
	}
	// if(($("select[name='input1Dropdown']").val() == "select")) {
	// 	check = "input1Dropdown";
	// }
	// if(($("select[name='input2Dropdown']").val() == "select")) {
	// 	check = "input2Dropdown";
	// }

	return check;
}

function checkInput(num) {
	let newElement = `<div id="errorBox${num}" class="errorBox"></div>`;
	$("#calcuateCoreContainer").append(newElement);
	$(`#errorBox${num}`).text(`Input ${num} cannot be empty`);
}

$("#input1").on('input', function() {
	if($(this).val() == "") {
		checkInput(1);
	}
	else {
		$("#errorBox1").remove();
	}
});

$("#input2").on('input', function() {
	if($(this).val() == "") {
		checkInput(2);
	}
	else {
		$("#errorBox2").remove();
	}
});

// $("#input1Dropdown").change(function() {
// 	$("#errorBoxInput1Dropdown").remove();
// });

// $("#input2Dropdown").change(function() {
// 	$("#errorBoxInput2Dropdown").remove();
// });


$("#calcuateCoreBtn").click(function () {
	let check = checkInputs();
	$("#errorBox").remove();

	if(check != "") {
		let newElement;
		switch(check) {
			case "select":
				newElement = `<div id="errorBox" class="errorBox"></div>`;
				$("#calcuateCoreContainer").append(newElement);
				$("#errorBox").text(`Select an operation`);
				break;
			case "input1":
				checkInput(1);
				break;
			case "input2":
				checkInput(2);
				break;
			// case "input1Dropdown":
			// 	newElement = `<div id="errorBoxInput1Dropdown" class="errorBox"></div>`;
			// 	$("#calcuateCoreContainer").append(newElement);
			// 	$("#errorBoxInput1Dropdown").text(`Data Type cannot be empty`);
			// 	break;
			// case "input2Dropdown":
			// 	newElement = `<div id="errorBoxInput2Dropdown" class="errorBox"></div>`;
			// 	$("#calcuateCoreContainer").append(newElement);
			// 	$("#errorBoxInput2Dropdown").text(`Data Type cannot be empty`);
			// 	break;
			default:
				null;
		}
	}
	else {
		let obj = createRequestObj();
		// $("#errorBoxInput1Dropdown").remove();
		// $("#errorBoxInput2Dropdown").remove();
		
		$.ajax({
			url: "http://127.0.0.1:8000/app/compute/",
			type: "POST",
			data: JSON.stringify(obj),
			contentType: "application/json",
			success: function(result) {
				$("#calcuateCoreBtn").text("Calculating");
				$("#calcuateCoreBtn").prop("disabled", true);
				$("#calcuateCoreResult").remove();
				let timer = setInterval(function() {
					pollResult(result.op_id, timer);
				}, 5000);
			},
			error: function(xhr, status, error) {
				console.log(xhr, status, error);
				let newElement = `<div id="errorBox" class="errorBox"></div>`;
				$("#calcuateCoreContainer").append(newElement);
				$("#errorBox").text(`${xhr.responseText}`);
			}
		});
	}
});

// Logistic regression


function pollLogisiticInputResult(id, timer) {
	$.ajax({
		url: `http://127.0.0.1:8000/app/status_logistic_regression/${id}`,
		success: function(result) {
			console.log(`${parseInt(result.percentage)}%`);
			$("#progressNumber").css("width", `${parseInt(result.percentage)}%`);
			$("#progressNumber").text(`${parseInt(result.percentage)}%`);
			if(parseInt(result.percentage) == 100) {
				let newElement = '<button id="trainLogisticInputBtn">Train</button>';
				$("#trainLogisticInputBtnContainer").append(newElement);
				clearInterval(timer);
				$("#predictLogisticInputBtn").prop("disabled", false);
			}
		},
		error: function(xhr, status, error) {
			console.log(xhr, status, error);
		}
	})
}

$("#trainLogisticInputBtn").click(function() {
	const reqObj = {
		data1: $("#inputLogisticInput1").val(),
		data2: $("#inputLogisticInput2").val(),
	}

	$.ajax({
		url: "http://127.0.0.1:8000/app/compute_logistic_regression/",
		type: "POST",
		data: JSON.stringify(reqObj),
		contentType: "application/json",
		success: function(result) {
			$("#trainLogisticInputBtn").remove();
			let progressBar = '<div id="progressNumber">0%</div>'
			$("#trainigProgressBar").append(progressBar);
			$("#progressNumber").css("width", `0%`);
			localStorage.setItem('logisticTrainId', result.id);
			let timer = setInterval(function() {
				pollLogisiticInputResult(result.id, timer);
			}, 5000);
		},
		error: function(xhr, status, error) {
			console.log(xhr, status, error);
		}
	})
});

$("#datasetLogistic").click(function() {
	$("#datasetLogisticContainer").css("display", "block");
	$("#inputLogisticContainer").css("display", "none");
	$("#datasetLogistic").addClass("active");
	$("#inputLogistic").removeClass("active");
});

$("#inputLogistic").click(function() {
	$("#datasetLogisticContainer").css("display", "none");
	$("#inputLogisticContainer").css("display", "block");
	$("#inputLogistic").addClass("active");
	$("#datasetLogistic").removeClass("active");
});

$("#predictLogisticInputBtn").click(function() {
	let obj = {
		data1: $("#predictLogisticInput").val()
	}

	$.ajax({
		url: `http://127.0.0.1:8000/app/predict_logistic_regression/${localStorage.getItem("logisticTrainId")}/`,
		type: "POST",
		data: JSON.stringify(obj),
		contentType: "application/json",
		success: function(result) {
			$("#outputPredictLogistic").text(JSON.stringify(result, null, 4));
			$("#outputPredictLogistic").css("padding", "20px");
			$("#outputPredictLogistic").css("border", "1px solid grey");
		},
		error: function(xhr, status, error) {
			console.log(xhr, status, error);
		}
	})
});

// Linear Regression stuff

$('input[type=radio][name=datasetSelect]').change(function() {
	switch(this.value) {
		case 'boston_house_prices':
		case 'breast_cancer':
			$("#chooseFileLinear").prop("disabled", true);
			$("#fileUploadBtnLinear").prop("disabled", true);
			break;
		case 'upload':
			$('input[type=radio][value=upload]').attr("disabled", false);
			$("#chooseFileLinear").prop("disabled", false);
			$("#fileUploadBtnLinear").prop("disabled", false);
			break;
		default:
			console.log('Nothing');
	}
});

$("input[id=chooseFileLinear]").change(function() {
	$(".uploadFileLabel").text($(this).val());
});

function setDatasetLinear() {
	$("#datasetLinearContainer").css("display", "block");
	$("#inputLinearContainer").css("display", "none");
	$("#datasetLinear").addClass("active");
	$("#inputLinear").removeClass("active");
}

function setInputLinear() {
	$("#datasetLinearContainer").css("display", "none");
	$("#inputLinearContainer").css("display", "block");
	$("#inputLinear").addClass("active");
	$("#datasetLinear").removeClass("active");
}

$("#datasetLinear").click(function() {
	setDatasetLinear();
});

$("#inputLinear").click(function() {
	setInputLinear();
});

function checkInputsLinear() {
	let check = ""

	if($("input[type=radio][name=upload]").is(':checked') && $("#chooseFileLinear")[0].files.length == 0) {
		check = "file";
	}
	
	if($("input[id=targetColumnLinear]").val() == "") {
		check = "tragetColumn";
	}

	return check;
}

function pollLinearResult(id, timer, progressElementId, newElementContainer, predictElementId, linearBtnConntainerId) {
	$.ajax({
		url: `http://127.0.0.1:8000/app/linear_regression/status/${id}/`,
		type: "GET",
		success: function(result) {
			console.log(`${parseInt(result.progress)}%`);
			$(`#${progressElementId}`).css("width", `${parseInt(result.progress)}%`);
			$(`#${progressElementId}`).text(`${parseInt(result.progress)}%`);
			if(parseInt(result.progress) == 100) {
				let newElement = `<button id="${newElementContainer}">Train</button>`;
				$(`#${linearBtnConntainerId}`).append(newElement);
				clearInterval(timer);
				$(`#${predictElementId}`).prop("disabled", false);
			}
		},
		error: function(xhr, status, error) {
			console.log(xhr, status, error);
		}
	});
}

$("#calcuateLinearBtn").click(function() {	
	let check = checkInputsLinear();

	if(check !== "") {
		let newElement;

		switch(check) {
			case "file":
				newElement = `<div id="errorBoxChecks" class="errorBox"></div>`;
				$("#calcuateLinearBtnContainer").append(newElement);
				$("#errorBoxChecks").text(`File not selected`);
				break;
			case "tragetColumn":
				newElement = `<div id="errorBoxChecks" class="errorBox"></div>`;
				$("#calcuateLinearBtnContainer").append(newElement);
				$("#errorBoxChecks").text(`Target Column cannot be empty`);
				break;
			default:
				null;
		}
	}
	else {
		let requestObj = new FormData();
		requestObj.append("target_column", $("input[id=targetColumnLinear]").val());
		if($("input[type=radio][value=upload]").is(':checked')) {
			requestObj.append("data_format", "file");
			requestObj.append("file", $("#chooseFileLinear")[0].files[0]);
		}
		else {
			requestObj.append("data_format", "dataset");
			console.log($("input[type=radio][name=datasetSelect]:checked").val());
			requestObj.append("dataset", $("input[type=radio][name=datasetSelect]:checked").val());
		}

		$.ajax({
			url: "http://127.0.0.1:8000/app/linear_regression/train/",
			data: requestObj,
			type: "POST",
			contentType: false,
			processData: false,
			success: function(result) {
				$("#errorBoxLinear").remove();
				$("#calcuateLinearBtn").remove();
				let progressBar = '<div id="progressNumberLinearDataset">0%</div>'
				$("#trainigProgressBarLinearDataset").append(progressBar);
				$("#progressNumberLinearDataset").css("width", `0%`);
				localStorage.setItem('linearTrainId', result.id);
				let timer = setInterval(function() {
					pollLinearResult(result.id, timer, "progressNumberLinearDataset", "calcuateLinearBtn", "predictLinearInputBtn", "calcuateLinearBtnContainer");
				}, 5000);
			},
			error: function(xhr, status, error) {
				console.log(xhr, status, error);
				let newElement = `<div id="errorBoxLinear" class="errorBox"></div>`;
				$("#calcuateLinearBtnContainer").append(newElement);
				$("#errorBox").text(`${xhr.responseText}`);
			}
		})
	}
});

function predictLinearRegression(id, outputContainerId) {
	$.ajax({
		url: `http://127.0.0.1:8000/app/linear_regression/predict/${id}/`,
		type: "POST",
		data: JSON.stringify(obj),
		contentType: "application/json",
		success: function(result) {
			$(`#${outputContainerId}`).text(JSON.stringify(result, null, 4));
			$(`#${outputContainerId}`).css("padding", "20px");
			$(`#${outputContainerId}`).css("border", "1px solid grey");
		},
		error: function(xhr, status, error) {
			console.log(xhr, status, error);
		}
	})
}

$("#predictLinearInputBtn").click(function() {
	let obj = {
		data1: $("#predictLinearInputDataset").val()
	}

	predictLinearRegression(localStorage.getItem("linearTrainId"), "outputPredictLinear");
});

// Linear Regression X&Y

function checkInputsLinearXY() {
	let check = "";
	
	if($("#inputLinearInput1").val() == "") {
		check = "input1";
	}
	if($("#inputLinearInput2").val() == "") {
		check = "input2";
	}

	return check;

}

$("#calcuateLinearBtnInput").click(function() {
	let check = checkInputsLinearXY();

	if(check !== "") {
		let newElement;

		switch(check) {
			case "input1":
				newElement = `<div id="errorBoxLinearXY" class="errorBox"></div>`;
				$("#calcuateLinearBtnContainerInput").append(newElement);
				$("#errorBoxLinearXY").text(`Enter the matrix X`);
				break;
			case "input2":
				newElement = `<div id="errorBoxLinearXY" class="errorBox"></div>`;
				$("#calcuateLinearBtnContainerInput").append(newElement);
				$("#errorBoxLinearXY").text(`Enter the matrix Y`);
				break;
			default:
				null;
		}
	}
	else {
		let requestObj = new FormData();
		requestObj.append("data_format", "matrices");
		requestObj.append("X", $("#inputLinearInput1").val());
		requestObj.append("y", $("#inputLinearInput2").val());

		$.ajax({
			url: "http://127.0.0.1:8000/app/linear_regression/train/",
			data: requestObj,
			type: "POST",
			contentType: false,
			processData: false,
			success: function(result) {
				$("#errorBoxLinearXY").remove();
				$("#calcuateLinearBtnInput").remove();
				let progressBar = '<div id="progressNumberLinearXY">0%</div>'
				$("#trainigProgressBarLinearInput").append(progressBar);
				$("#progressNumberLinearXY").css("width", `0%`);
				localStorage.setItem('linearTrainId', result.id);
				let timer = setInterval(function() {
					pollLinearResult(result.id, timer, "progressNumberLinearXY", "calcuateLinearBtnInput", "predictLinearBtnInput", "calcuateLinearBtnContainerInput");
				}, 5000);
			},
			error: function(xhr, status, error) {
				console.log(xhr, status, error);
				let newElement = `<div id="errorBoxLinearXY" class="errorBox"></div>`;
				$("#calcuateLinearBtnContainerInput").append(newElement);
				$("#errorBoxLinearXY").text(`${error}`);
			}
		})
	}
});

$("#predictLinearBtnInput").click(function() {
	let obj = {
		data1: $("#predictLinearInputXY").val()
	}

	predictLinearRegression(localStorage.getItem("linearTrainId"), "outputPredictLinearXY");
});