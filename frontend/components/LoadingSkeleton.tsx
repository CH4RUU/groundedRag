import styles from "./LoadingSkeleton.module.css";

export default function LoadingSkeleton() {
  return (
    <div className={`glass ${styles.skeleton}`} aria-label="Loading response" role="status">
      <div className={styles.metaBar}>
        <div className={`${styles.shimmer} ${styles.badgePH}`} />
        <div className={`${styles.shimmer} ${styles.barPH}`} />
      </div>
      <div className={styles.body}>
        <div className={`${styles.shimmer} ${styles.line} ${styles.long}`} />
        <div className={`${styles.shimmer} ${styles.line} ${styles.medium}`} />
        <div className={`${styles.shimmer} ${styles.line} ${styles.short}`} />
        <div className={`${styles.shimmer} ${styles.line} ${styles.long}`} style={{ marginTop: 16 }} />
        <div className={`${styles.shimmer} ${styles.line} ${styles.medium}`} />
      </div>
      <div className={styles.footer}>
        {[1, 2, 3].map((i) => (
          <div key={i} className={`${styles.shimmer} ${styles.chipPH}`} />
        ))}
      </div>
    </div>
  );
}
