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

function pollResult(opId, timer) {
	$.ajax({
		url: `http://127.0.0.1:8000/app/result/${opId}`,
		type: "GET",
		success: function(res) {
			if(res.status == "computed") {
				$("#calcuateCoreResult").text(`Result: ${res.result}`);
				clearInterval(timer);
			}
		}
	});
}


$("#calcuateCoreBtn").click(function () {
	let obj = {
		data1: parseInt($("#input1").val()),
		data2: parseInt($("#input2").val()),
		type1: "integer",
		type2: "integer"
	};

	$.ajax({
		url: "http://127.0.0.1:8000/app/compute/",
		type: "POST",
		data: JSON.stringify(obj),
		contentType: "application/json",
		success: function(result) {
			let newElement = `<div id="calcuateCoreResult">Caclulating</div>`
			$("#calcuateCoreBtn").remove();
			$("#calcuateCoreContainer").append(newElement);
			let timer = setInterval(function() {
				pollResult(result.op_id, timer);
			}, 5000);
		}
	});
});