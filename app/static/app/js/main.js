$(document).ready(function () {
	$("#core").addClass('active');
	setDisplayNone();
	$("#coreContainer").css("display", "flex");

	$("input[id=chooseFile]").change(function() {
		$(".uploadFileLabel").text($(this).val());
	});
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
});

$("#logistic").click(function () {
	removeClasses();
	setDisplayNone();
	$("#logistic").addClass('active');
	$("#logisticContainer").css("display", "flex");
});

$("select[name='coreDropdown']").change(function() {
	if($(this).val() >= 6) {
		$("#input2").prop("disabled", true);
		$("#input2Dropdown").prop("disabled", true);
	}
	else {
		$("#input2").prop("disabled", false);
		$("#input2Dropdown").prop("disabled", false);
	}
})

function pollResult(opId, timer) {
	$.ajax({
		url: `http://127.0.0.1:8000/app/result/${opId}`,
		type: "GET",
		success: function(res) {
			if(res.status == "computed") {
				let newElement = `<div id="calcuateCoreResult"></div>`;
				$("#calcuateCoreContainer").append(newElement);
				$("#calcuateCoreResult").text(`Result: ${res.result}`);
				$("#calcuateCoreBtn").text("Calculate");
				clearInterval(timer);
			}
		}
	});
}

function createRequestObj() {
	let obj = {
		operation: parseInt($("select[name='coreDropdown']").val()),
		data1: $("#input1").val(),
	};

	if($("select[name='coreDropdown']").val() >= 6) {
		obj["data2"] = null;
		
	}
	else {
		obj["data2"] = $("#input2").val();

		if($("select[name='input1Dropdown']").val() === "matrix" && $("select[name='input2Dropdown']").val() === "matrix") {
			obj.operation = 1;
		}
	}

	return obj;
}


$("#calcuateCoreBtn").click(function () {
	let obj = createRequestObj();

	console.log(obj);
	$.ajax({
		url: "http://127.0.0.1:8000/app/compute/",
		type: "POST",
		data: JSON.stringify(obj),
		contentType: "application/json",
		success: function(result) {
			$("#calcuateCoreBtn").text("Calculating");
			$("#calcuateCoreResult").remove();
			let timer = setInterval(function() {
				pollResult(result.op_id, timer);
			}, 5000);
		}
	});
});