document.addEventListener('DOMContentLoaded', () => {

const figureDialog = document.querySelector('.research-lightbox');
document.querySelectorAll('.research-figure').forEach(button => button.addEventListener('click', () => {
  const source = button.querySelector('img');
  const image = figureDialog.querySelector('img');
  image.src = source.src; image.alt = source.alt;
  figureDialog.querySelector('p').textContent = source.alt;
  figureDialog.showModal();
}));
figureDialog.querySelector('button').addEventListener('click', () => figureDialog.close());
figureDialog.addEventListener('click', event => {if(event.target === figureDialog) figureDialog.close();});

});
