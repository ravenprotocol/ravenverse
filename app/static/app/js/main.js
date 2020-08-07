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