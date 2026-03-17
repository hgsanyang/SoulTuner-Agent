import React from 'react';
import styles from './StarryBackground.module.css';

export default function StarryBackground() {
  return (
    <div className={styles.starryBg}>
      <div className={styles.galaxy}></div>
      <div className={styles.galaxy2}></div>
      <div className={styles.stars}></div>
      <div className={styles.mediumStars}></div>
      <div className={`${styles.shootingStar} ${styles.star1}`}></div>
      <div className={`${styles.shootingStar} ${styles.star2}`}></div>
      <div className={`${styles.shootingStar} ${styles.star3}`}></div>
      <div className={`${styles.shootingStar} ${styles.star4}`}></div>
      <div className={`${styles.shootingStar} ${styles.star5}`}></div>
    </div>
  );
}
