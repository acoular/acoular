
function toggle_row(name) {
    /* compensate for MS's failure to comply with CSS standard */
    var tr_display = 'table-row';
    if (navigator.appName == "Microsoft Internet Explorer") {
        tr_display = 'block';
    }
    tr = document.getElementById(name + '-tree');
    td = document.getElementById(name + '-icon');
    text = td.firstChild;
    table = tr.parentNode;
    /*img = table.getElementsByTagName('img')[0];*/
    if (!tr.style.display || (tr.style.display == 'none')) {
        /* toggle on */
        tr.style.display = tr_display;
        /*img.src = 'minus.png';*/
        text.replaceData(1, 1, '-');
    } else {
        /* toggle off */
        tr.style.display = 'none';
        /*img.src = 'plus.png';*/
        text.replaceData(1, 1, '+');
    }
}
